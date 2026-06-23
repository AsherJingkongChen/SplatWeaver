from dataclasses import dataclass
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from jaxtyping import Float
from torch import Tensor
import torch

from .loss import Loss


def wavelet_high_freq_map(
    imgs: torch.Tensor,  # [B, V, C, H, W], in [0,1]
    wave: str = "db2"
):
    """
    Gray-scale wavelet high-frequency energy map.

    Returns:
        high_freq_map: [B, V, C, H, W]  (gray processed, then broadcast)
    """
    assert imgs.dim() == 5, "Input must be BVCHW"
    assert imgs.shape[2] == 3, "Expect RGB input"

    B, V, C, H, W = imgs.shape
    device = imgs.device

    # Y = 0.299 R + 0.587 G + 0.114 B
    r, g, b = imgs[:, :, 0], imgs[:, :, 1], imgs[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b  # [B, V, H, W]

    gray = gray.unsqueeze(2)  # [B, V, 1, H, W]

    x = gray.view(B * V, 1, H, W)

    dwt = DWTForward(
        J=1,
        wave=wave,
        mode="symmetric"
    ).to(device)

    yl, yh = dwt(x)

    LH = yh[0][:, :, 0]
    HL = yh[0][:, :, 1]
    HH = yh[0][:, :, 2]

    high_freq_energy = torch.sqrt(LH.pow(2) + HL.pow(2) + HH.pow(2) + 1e-8)  # [BV, 1, H/2, W/2]

    high_freq_energy = F.interpolate(
        high_freq_energy,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )  # [BV, 1, H, W]

    high_freq_energy = high_freq_energy.view(B, V, 1, H, W)

    return high_freq_energy


def build_frequency_target(freq_prior):
    """
    freq_prior: [B, V, 1, H, W]
    return:
        target_expert: [B, V, H, W]  values in {0,1,2,3}
    """
    B, V, _, H, W = freq_prior.shape

    freq = freq_prior.view(B, -1)  # [B, V*H*W]

    # compute quantiles per batch
    p3 = torch.quantile(freq, 0.98, dim=1, keepdim=True)
    p2 = torch.quantile(freq, 0.96, dim=1, keepdim=True)
    p1 = torch.quantile(freq, 0.76, dim=1, keepdim=True)

    freq_full = freq_prior.view(B, V * H * W)

    target = torch.zeros_like(freq_full, dtype=torch.long)

    target[freq_full >= p3] = 3
    target[(freq_full < p3) & (freq_full >= p2)] = 2
    target[(freq_full < p2) & (freq_full >= p1)] = 1
    target[freq_full < p1] = 0

    target = target.view(B, V, H, W)
    return target



def LossExpert(choices, expert_soft, context_gt, loss_weight=0.01):

    B, V, E, H, W = choices.shape
    with torch.no_grad():
        high_fre_map = wavelet_high_freq_map(context_gt)
        target = build_frequency_target(high_fre_map)

    expert_soft = expert_soft.permute(0, 1, 3, 4, 2).reshape(-1, E)
    target = target.reshape(-1)

    loss = F.cross_entropy(
    expert_soft, 
    target, 
    label_smoothing=0.1,
    reduction='mean'
)

    one = choices[:, :, 1, :, :]
    two = choices[:, :, 2, :, :]
    three = choices[:, :, 3, :, :]

    return ( loss+ ((one.mean()-0.2)**2 + (two.mean()-0.02)**2 + (three.mean()-0.02)**2)*5) * loss_weight


