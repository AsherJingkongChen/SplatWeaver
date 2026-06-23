import os
from pathlib import Path
import sys
import json
import gzip
import argparse
import numpy as np
from PIL import Image
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
import torch
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splatweaver.model.ply_export import export_ply
from splatweaver.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from misc.image_io import save_image, save_interpolated_video
from splatweaver.utils.image import process_image, process_image_448,process_image_252, visualize_expert_selection
from omegaconf import DictConfig, OmegaConf
from splatweaver.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from splatweaver.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from splatweaver.model.encoder.splatweaver import EncoderSplatWeaver, EncoderSplatWeaverCfg, OpacityMappingCfg
import warnings
from dacite import from_dict, Config
warnings.filterwarnings("ignore")
from splatweaver.model.model import get_model
from splatweaver.model.model.splatweaver import SplatWeaver
from splatweaver.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
dacite_config = Config(cast=[tuple])

def setup_args():
    """Set up command-line arguments for the eval NVS script."""
    parser = argparse.ArgumentParser(description='Test SplatWeaver on NVS evaluation')
    parser.add_argument('--data_dir', type=str, default="/", help='Path to NVS dataset')
    parser.add_argument('--llffhold', type=int, default=8, help='LLFF holdout')
    parser.add_argument('--num_views', type=int, default=16, help='Number of context views')
    parser.add_argument('--ckpt_path', type=str, default="X.ckpt", help='Path to ckpt path')
    parser.add_argument('--output_path', type=str, default="outputs/test/", help='Path to output directory')
    return parser.parse_args()

def compute_metrics(pred_image, image):
    psnr = compute_psnr(pred_image, image)
    ssim = compute_ssim(pred_image, image)
    lpips = compute_lpips(pred_image, image)
    return psnr, ssim, lpips

def remove_model_prefix(state_dict):
    return {k.replace('model.', '', 1): v for k, v in state_dict.items()}

def load_model_from_json(config_path: str):
    with open(config_path, "r") as f:
        raw_cfg = json.load(f)

    encoder_cfg = from_dict(
        data_class=EncoderSplatWeaverCfg,
        data=raw_cfg["encoder_cfg"],
        config=dacite_config
    )
    decoder_cfg = from_dict(
        data_class=DecoderSplattingCUDACfg,
        data=raw_cfg["decoder_cfg"],
        config=dacite_config
    )

    model = get_model(encoder_cfg=encoder_cfg, decoder_cfg=decoder_cfg)
    return model

def uniform_indices_py(N, K):
    return [int(i * (N-1) / (K-1)) for i in range(K)]

def evaluate(args: argparse.Namespace):
    model = load_model_from_json("config/config.json")
    model.load_state_dict(remove_model_prefix(torch.load(args.ckpt_path)["state_dict"]), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    os.makedirs(args.output_path, exist_ok=True)
    # jsonl output
    jsonl_path = os.path.join(args.output_path, "scene_metrics.jsonl")
    jsonl_file = open(jsonl_path, "w", encoding="utf-8")

    image_folders = os.listdir(args.data_dir)
    image_folders.sort()
    psnr_t = []
    ssim_t= []
    lpips_t = []
    for folder in image_folders:

        image_folder = os.path.join(args.data_dir,folder)

        image_names = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
        images = [process_image(img_path) for img_path in image_names]
        ctx_indices = uniform_indices_py(len(image_names),args.num_views)
        tgt_indices = [idx for idx, name in enumerate(image_names) if (idx+1) % args.llffhold == 0]

        ctx_images = torch.stack([images[i] for i in ctx_indices], dim=0).unsqueeze(0).to(device)
        tgt_images = torch.stack([images[i] for i in tgt_indices], dim=0).unsqueeze(0).to(device)
        ctx_images = (ctx_images+1)*0.5
        tgt_images = (tgt_images+1)*0.5
        b, v, _, h, w = tgt_images.shape

        # run inference

        encoder_output = model.encoder(
            ctx_images,
            global_step=0,
            visualization_dump={},
        )
        
        gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose
        choice_one = encoder_output.choice[0][1]

        num_context_view = ctx_images.shape[1]
        vggt_input_image = torch.cat((ctx_images, tgt_images), dim=1).to(torch.bfloat16)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            aggregated_tokens_list, patch_start_idx = model.encoder.aggregator(vggt_input_image, intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx)
        with torch.cuda.amp.autocast(enabled=False):
            fp32_tokens = [token.float() for token in aggregated_tokens_list]
            pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
            pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(pred_all_pose_enc, vggt_input_image.shape[-2:])

        extrinsic_padding = torch.tensor([0, 0, 0, 1], device=pred_all_extrinsic.device, dtype=pred_all_extrinsic.dtype).view(1, 1, 1, 4).repeat(b, vggt_input_image.shape[1], 1, 1)
        pred_all_extrinsic = torch.cat([pred_all_extrinsic, extrinsic_padding], dim=2).inverse()

        pred_all_intrinsic[:, :, 0] = pred_all_intrinsic[:, :, 0] / w
        pred_all_intrinsic[:, :, 1] = pred_all_intrinsic[:, :, 1] / h
        pred_all_context_extrinsic, pred_all_target_extrinsic = pred_all_extrinsic[:, :num_context_view], pred_all_extrinsic[:, num_context_view:]
        pred_all_context_intrinsic, pred_all_target_intrinsic = pred_all_intrinsic[:, :num_context_view], pred_all_intrinsic[:, num_context_view:]

        scale_factor = pred_context_pose['extrinsic'][:, :, :3, 3].mean() / pred_all_context_extrinsic[:, :, :3, 3].mean()
        pred_all_target_extrinsic[..., :3, 3] = pred_all_target_extrinsic[..., :3, 3] * scale_factor
        pred_all_context_extrinsic[..., :3, 3] = pred_all_context_extrinsic[..., :3, 3] * scale_factor
        print("scale_factor:", scale_factor)
        output = model.decoder.forward(
            gaussians,
            pred_all_target_extrinsic,
            pred_all_target_intrinsic.float(),
            torch.ones(1, v, device=device) * 0.01,
            torch.ones(1, v, device=device) * 100,
            (h,w)
            )

        save_interpolated_video(pred_all_context_extrinsic, pred_all_context_intrinsic, b, h, w, gaussians, args.output_path+folder, model.decoder)
        # Save original images
        save_path = Path(args.output_path+folder)
        os.makedirs(save_path, exist_ok=True)
        visualize_expert_selection(choice_one, save_path=args.output_path+folder+"/expert_selection.png")
        
        # compute metrics
        psnr, ssim, lpips = compute_metrics(output.color[0], tgt_images[0])
        psnr_mean = psnr.mean().cpu().numpy()
        ssim_mean = ssim.mean().cpu().numpy()
        lpips_mean = lpips.mean().cpu().numpy()

        print(f"Scene: {folder}, \n PSNR: {psnr_mean:.2f}, SSIM: {ssim_mean:.3f}, LPIPS: {lpips_mean:.3f}")

        for idx, (gt_image, pred_image) in enumerate(zip(tgt_images[0], output.color[0])):
            save_image(gt_image, save_path / "gt" / f"{idx:0>6}.jpg")
            save_image(pred_image, save_path / "pred" / f"{idx:0>6}.jpg")

        json_line = {
            "Scene": folder,
            "PSNR": float(f"{psnr_mean:.2f}"),
            "SSIM": float(f"{ssim_mean:.3f}"),
            "LPIPS": float(f"{lpips_mean:.3f}"),
            "GS": gaussians.opacities.shape[1]
        }
        jsonl_file.write(json.dumps(json_line) + "\n")

        psnr_t.append(psnr_mean)
        ssim_t.append(ssim_mean)
        lpips_t.append(lpips_mean)

    jsonl_file.close()

    print(f"PSNR: {np.mean(psnr_t):.2f}, SSIM: {np.mean(ssim_t):.3f}, LPIPS: {np.mean(lpips_t):.3f}")
    #export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], save_path / "gaussians.ply")

if __name__ == "__main__":
    args = setup_args()
    evaluate(args)
