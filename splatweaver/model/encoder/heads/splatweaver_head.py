from einops import rearrange
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
# import dust3r.utils.path_to_croco
from .dpt_block import DPTOutputAdapter, Interpolate, make_fusion_block
from splatweaver.model.encoder.vggt.heads.dpt_head import DPTHead
from .head_modules import UnetExtractor, AppearanceTransformer, _init_weights
from .postprocess import postprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import numpy as np
import time
import faiss
import torch
from faiss.contrib import torch_utils

def wavelet_high_freq_map(
    imgs: torch.Tensor,  # [B, V, C, H, W], in [0,1]
    wave: str = "db2"
):
    """
    Gray-scale wavelet high-frequency energy map.

    Returns:
        high_freq_map: [B, V, C, H, W]  (gray processed, then broadcast)
    """
    assert imgs.dim() == 4, "Input must be BVCHW"
    assert imgs.shape[1] == 3, "Expect RGB input"

    V, C, H, W = imgs.shape
    device = imgs.device

    # Y = 0.299 R + 0.587 G + 0.114 B
    r, g, b = imgs[:, 0], imgs[:, 1], imgs[:, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b  # [V, H, W]

    gray = gray.unsqueeze(1)  # [V, 1, H, W]
    x = gray
    dwt = DWTForward(
        J=1,
        wave=wave,
        mode="symmetric"
    ).to(device)

    yl, yh = dwt(x)
    yh_flat = yh[0].squeeze(1)

    wavelet_coeffs = yh_flat

    return wavelet_coeffs


_GPU_RESOURCES = {}

def get_gpu_resource():
    """Get or create StandardGpuResources for current GPU."""
    current_device = torch.cuda.current_device()
    if current_device not in _GPU_RESOURCES:
        # Ensure we're on the right GPU
        torch.cuda.set_device(current_device)
        res = faiss.StandardGpuResources()
        #res.setTempMemory(256 * 1024 * 1024)  # 256MB temp memory
        _GPU_RESOURCES[current_device] = res
    return _GPU_RESOURCES[current_device]

@torch.no_grad()
def knn_faiss(x: torch.Tensor, k: int, nlist: int = None, nprobe: int = 4):
    """
    FAISS IVF approximate KNN on GPU (for large N).

    Args:
        x: [N, D] tensor on GPU (local to current DDP process)
        k: number of neighbors (excluding self)
        nlist: number of clusters (default: max(100, N//1000))
        nprobe: number of clusters to search (default: 8)

    Returns:
        indices: [N, k] LongTensor on same device as x
    """
    assert x.is_cuda, "Input must be on GPU"
    N, D = x.shape

    if N <= k:
        torch.rand(908016, 38)
        print(f"Number of points ({N}) <= k ({k})")
        #raise ValueError(f"Number of points ({N}) <= k ({k})")

    if nlist is None:
        nlist =  N // 10000
    nlist = min(nlist, N)  # cannot have more clusters than points

    x_np = x.cpu().numpy()  # [N, D]

    res = get_gpu_resource()
    current_device = torch.cuda.current_device()
    quantizer = faiss.IndexFlatL2(D)

    cpu_index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)
    cpu_index.train(x_np)

    gpu_index = faiss.index_cpu_to_gpu(res, current_device, cpu_index)
    gpu_index.nprobe = min(nprobe, nlist)

    gpu_index.add(x_np)
    _, idx = gpu_index.search(x_np, k + 1)

    idx = np.where(idx == -1, np.arange(N).reshape(-1, 1), idx)

    idx_torch = torch.from_numpy(idx).to(x.device, non_blocking=True)
    return idx_torch

class NeighborFusionLayer(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x, pos, knn_idx):
        """
        x: (N, C)
        pos: (N, 3)
        knn_idx: (N, k)
        """
        q = self.to_q(x)                   # (N, C)
        k = self.to_k(x[knn_idx])          # (N, k, C)
        v = self.to_v(x[knn_idx])          # (N, k, C)

        rel_pos = pos[knn_idx] - pos.unsqueeze(1)
        rel_feat = self.pos_mlp(rel_pos)   # (N, k, C)

        attn_input = q.unsqueeze(1) - k + rel_feat
        attn_weights = self.attn_mlp(attn_input) # (N, k, C)
        attn_weights = self.softmax(attn_weights)

        out = torch.sum(attn_weights * (v + rel_feat), dim=1) # (N, C)
        return out


class GaussianParamPredictor(nn.Module):
    def __init__(self, k=8, in_dim=128, out_dim=27+7+1, backend='faiss', hidden_dim=32):
        super().__init__()
        self.k = k
        self.backend = backend
        self.out_dim = out_dim
        self.input = nn.Linear(in_dim, hidden_dim)

        self.pt_block = NeighborFusionLayer(hidden_dim, k)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, out_dim)
        )

    def knn(self, X):
        pos = X[:, :3]
        if self.backend == 'faiss':
            return knn_faiss(pos, self.k)

    def forward(self, X):
        pos = X[:, :3]
        knn_idx = self.knn(X)
        feat = self.pt_block(self.input(X), pos, knn_idx)
        return pos, self.head(feat)+X[:, 3:3+self.out_dim]


class CardinalityGaussian(nn.Module):
    def __init__(self, in_ch, hidden=128, base_slot_dim=14, n_experts=4, xyz_bias=0.01, temp=0.5):
        super().__init__()
        self.in_ch = in_ch
        self.hidden = hidden
        self.base_slot_dim = base_slot_dim
        self.n_experts = n_experts
        self.temp = temp
        self.xyz_bias = xyz_bias
        self.expert_out_dims = [0] + [k * base_slot_dim for k in range(1, n_experts)]

        self.gating = nn.Sequential(
            nn.Conv2d(in_ch, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_experts, kernel_size=1)
        )
        self.pixel_projector = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32)
        )
        self.gs_predictor = GaussianParamPredictor(k=8, in_dim=base_slot_dim+32, out_dim=base_slot_dim-3)

        with torch.no_grad():
            final_conv = self.gating[-1]
            final_conv.bias.zero_()
            final_conv.bias[1] = 2.0
            final_conv.bias[0] = 2.5

        self.experts = nn.ModuleList()
        for out_dim in self.expert_out_dims:
            if out_dim == 0:
                self.experts.append(nn.Identity())
            else:
                self.experts.append(nn.Sequential(
                    nn.Linear(in_ch, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, out_dim)
                ))

    def forward(self, feats,feats_router, points, temperature=None, hard=True, deterministic=False):
        """
        feats: [B, C, H, W]
        points: [B, H, W, 3]
        temperature: float or None
        hard: if True, use hard one-hot selection (via gumbel_softmax hard)
        deterministic: if True, use argmax selection (no gumbel noise)
        Returns:
            slots: tensor (M, base_slot_dim) concatenated
            meta: dict with mapping info to reconstruct per-pixel slots
        """
        B, C, H, W = feats.shape
        N = B * H * W
        device = feats.device
        temperature = self.temp if temperature is None else temperature

        logits = self.gating(feats_router)  # [B, E, H, W]
        logits = logits.permute(0, 2, 3, 1).reshape(-1, self.n_experts)  # [N, E]

        if deterministic:
            choices = torch.zeros_like(logits)
            idx = logits.argmax(dim=-1, keepdim=True)
            choices.scatter_(-1, idx, 1.0)
        else:
            choices = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)  # [N, E]

        if hard or deterministic:
            assigned = choices.argmax(dim=-1)  # [N]
        else:
            assigned = choices.argmax(dim=-1)

        feats_flat = feats.permute(0, 2, 3, 1).reshape(N, C)
        points_flat = points.reshape(N, 3)

        slots_list = []  # will collect tensors of shape [num_slots, base_slot_dim]

        for e, out_dim in enumerate(self.expert_out_dims):
            mask = (assigned == e)
            idxs = mask.nonzero(as_tuple=False).squeeze(-1)  # indices of pixels assigned to expert e
            print('Export: ',e, 'Number of pixels: ',len(idxs))
            if idxs.numel() == 0 or out_dim == 0:
                # no pixels or no-op expert -> skip (no slots created)
                continue
            # gather pixel features -> [P, C]
            pf = feats_flat.index_select(0, idxs)
            points_select = points_flat.index_select(0, idxs)
            # run expert network -> [P, out_dim]
            expert_net = self.experts[e]
            out = expert_net(pf) * choices.index_select(0, idxs)[:,e:e+1] # [P, out_dim]
            # split into base_slot_dim chunks along last dim
            num_chunks = out_dim // self.base_slot_dim
            out = out.view(-1, num_chunks, self.base_slot_dim)  # [P, K, base]
            # flatten chunks into slots -> [P*K, base]
            P = out.shape[0]
            out_slots = out.reshape(P * num_chunks,self.base_slot_dim)

            xyz_part = out_slots[:, :3]
            other_part = out_slots[:, 3:]
            xyz_processed = torch.tanh(xyz_part) * self.xyz_bias + points_select.repeat_interleave(num_chunks, dim=0)

            pixel_feature = self.pixel_projector(pf).repeat_interleave(num_chunks, dim=0)
            out_slots = torch.cat([xyz_processed, other_part, pixel_feature], dim=1)

            slots_list.append(out_slots)

        if len(slots_list) == 0:
            # nothing routed -> return empty tensor
            slots_final = feats.new_empty((0, self.base_slot_dim))
        else:
            slots_final = torch.cat(slots_list, dim=0)
        slots_final_pos, slots_final_param = self.gs_predictor(slots_final)
        print('Number of pixels: ', N, 'Number of predicted: ', slots_final.shape[0], 'Number of final: ', slots_final_param.shape[0])

        return slots_final_pos, slots_final_param, choices.view(B, H, W, self.n_experts).permute(0,3,1,2).contiguous(), logits.view(B, H, W, self.n_experts).permute(0,3,1,2).contiguous()


class SplatWeaver_Head(DPTHead):
    def __init__(self,
            dim_in: int,
            patch_size: int = 14,
            output_dim: int = 83,
            activation: str = "inv_log",
            conf_activation: str = "expp1",
            features: int = 256,
            out_channels: List[int] = [256, 512, 1024, 1024],
            intermediate_layer_idx: List[int] = [4, 11, 17, 23],
            pos_embed: bool = True,
            feature_only: bool = False,
            down_ratio: int = 1,
    ):
        super().__init__(dim_in, patch_size, output_dim, activation, conf_activation, features, out_channels, intermediate_layer_idx, pos_embed, feature_only, down_ratio)

        head_features_1 = 128
        head_features_2 = 128

        self.input_merger = nn.Sequential(
            nn.Conv2d(3, 32, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(32, head_features_1, 5, 1, 2),
            nn.ReLU(),
        )
        self.wave_merger1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.Conv2d(32, 32, 1, 1),
        )
        self.wave_merger2 = nn.Sequential(
            nn.Conv2d(32, 128, 3, 1, 1),
            nn.Sigmoid()
        )


        self.gs_moe = CardinalityGaussian(in_ch=head_features_2, hidden=head_features_2, base_slot_dim=output_dim, xyz_bias=0.005)

        self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, encoder_tokens: List[torch.Tensor], points, imgs, patch_start_idx: int = 5, image_size=None, conf=None, frames_chunk_size: int = 8):
        B, S, _, H, W = imgs.shape
        image_size = self.image_size if image_size is None else image_size

        pts_list = []
        feat_list = []
        choice_list = []
        logit_list = []
        feat_dense_list = []
        for B_i in range(B):
            pts_bi, feat_bi, choice_i, logit_i, out1_i = self._forward_impl(B_i, points[B_i],
                encoder_tokens, imgs[B_i], patch_start_idx
            )
            pts_list.append(pts_bi)
            feat_list.append(feat_bi)
            choice_list.append(choice_i)
            logit_list.append(logit_i)
            feat_dense_list.append(out1_i)

        return (
            pts_list,
            feat_list,
            torch.stack(choice_list, dim=0),
            torch.stack(logit_list, dim=0),
            torch.stack(feat_dense_list, dim=0),
        )

    def _forward_impl(self,B_i, points, encoder_tokens: List[torch.Tensor], imgs, patch_start_idx: int = 5, frames_start_idx: int = None, frames_end_idx: int = None):
        # points : # V, H, W, 3

        S, _, H, W = imgs.shape

        patch_h, patch_w = H // self.patch_size[0], W // self.patch_size[1]

        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            # x = encoder_tokens[layer_idx][:, :, patch_start_idx:]
            if len(encoder_tokens) > 10:
                x = encoder_tokens[layer_idx][B_i, :, patch_start_idx:]
            else:
                list_idx = self.intermediate_layer_idx.index(layer_idx)
                x = encoder_tokens[list_idx][B_i, :, patch_start_idx:]


            x = x.view(S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        direct_img_feat = self.input_merger(imgs)
        wave_tensor = wavelet_high_freq_map(imgs)
        wave_feat = self.wave_merger1(wave_tensor)
        wave_feat =  F.interpolate(
        wave_feat,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )
        wave_feat = self.wave_merger2(wave_feat)

        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        out = out + direct_img_feat

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        out1 = self.scratch.output_conv2(out)
        out2 = out1 + out*wave_feat
        pts, feat, choice, logit = self.gs_moe(out1,out2,points)

        return pts, feat, choice, logit, out1

