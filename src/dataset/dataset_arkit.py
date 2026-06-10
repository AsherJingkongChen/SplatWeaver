from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal
import os
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization


@dataclass
class DatasetArkitCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    avg_pose: bool
    rescale_to_1cube: bool
    intr_augment: bool
    normalize_by_pts3d: bool
    rescale_to_1cube: bool


@dataclass
class DatasetArkitCfgWrapper:
    arkit: DatasetArkitCfg

def get_subfolders_os(data_root):
    subfolders = []
    for entry in os.listdir(data_root):
        full_path = os.path.join(data_root, entry)
        if os.path.isdir(full_path):
            subfolders.append(entry)
    return subfolders

class DatasetArkit(Dataset):
    cfg: DatasetArkitCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0
    
    def __init__(
        self,
        cfg: DatasetArkitCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        
        # load data
        self.data_root = cfg.roots[0]
        self.data_list = []
        
        data_index = get_subfolders_os(self.data_root)
        
        self.data_list = [
            os.path.join(self.data_root, item) for item in data_index
        ] 
        
        self.scene_ids = {}
        self.scenes = {}
        index = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(self.load_jsons, scene_path) for scene_path in self.data_list]
            for future in as_completed(futures):
                scene_frames, scene_id = future.result()
                self.scenes[scene_id] = scene_frames
                self.scene_ids[index] = scene_id
                index += 1
        print(f"Arkit Dataset: {self.stage}: loaded {len(self.scene_ids)} scenes")
        

    def load_jsons(self, scene_path):

        scene_frames = []
        scene_id = scene_path.split("/")[-1].split(".")[0]
        for i, image_path in enumerate(os.listdir(scene_path)):
            frame_tmp = {}
            frame_tmp["file_path"] = os.path.join(scene_path, image_path)
            scene_frames.append(frame_tmp)
        return scene_frames, scene_id

    def load_frames(self, frames):
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Create a list to store futures with their original indices
            futures_with_idx = []
            for idx, file_path in enumerate(frames):
                file_path = file_path["file_path"]
                futures_with_idx.append(
                    (
                        idx,
                        executor.submit(
                            lambda p: self.to_tensor(Image.open(p).convert("RGB")),
                            file_path,
                        ),
                    )
                )
            
            # Pre-allocate list with correct size to maintain order
            torch_images = [None] * len(frames)
            for idx, future in futures_with_idx:
                torch_images[idx] = future.result()
            # Check if all images have the same size
            sizes = set(img.shape for img in torch_images)
            if len(sizes) == 1:
                torch_images = torch.stack(torch_images)
        # Return as list if images have different sizes
        return torch_images

        
    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]
        
    def getitem(self, index: int, num_context_views: int, patchsize: tuple) -> dict:
        
        scene = self.scene_ids[index]
        
        example = self.scenes[scene]
        extrinsics = torch.rand(len(example), 4, 4, dtype=torch.float32)
        intrinsics = torch.rand(len(example), 3, 3, dtype=torch.float32)
        try:
            context_indices, target_indices, overlap = self.view_sampler.sample(
                scene,
                num_context_views,
                extrinsics,
                intrinsics,
            )
        except ValueError:
            # Skip because the example doesn't have enough frames.
            raise Exception("Not enough frames")
        

        input_frames = [example[i] for i in context_indices]
        target_frame = [example[i] for i in target_indices]
        
        context_images = self.load_frames(input_frames)
        target_images = self.load_frames(target_frame)

      
        context_depth = torch.ones_like(context_images)[:, 0]
        target_depth = torch.ones_like(target_images)[:, 0]

        scale = 1

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "depth": context_depth,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "depth": target_depth,
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,
            },
            "scene": "general_"+scene,
        }
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        if self.stage == "train" and self.cfg.intr_augment:
            intr_aug = True
        else:
            intr_aug = False
        
        example = apply_crop_shim(example, (patchsize[0] * 14, patchsize[1] * 14), intr_aug=intr_aug)

        context_pts3d = torch.ones_like(example["context"]["image"]).permute(0, 2, 3, 1) # [N, H, W, 3]
        context_valid_mask = torch.ones_like(example["context"]["image"])[:, 0].bool() # [N, H, W]

        target_pts3d = torch.ones_like(target_images).permute(0, 2, 3, 1) # [N, H, W, 3]
        target_valid_mask = torch.ones_like(target_images)[:, 0].bool() # [N, H, W]
        

        example["context"]["pts3d"] = context_pts3d
        example["target"]["pts3d"] = target_pts3d
        example["context"]["valid_mask"] = context_valid_mask * -1
        example["target"]["valid_mask"] = target_valid_mask * -1

        return example
        
    def __getitem__(self, index_tuple: tuple) -> dict:
        index, num_context_views, patchsize_h = index_tuple
        patchsize_w = (self.cfg.input_image_shape[1] // 14)
        try:
            return self.getitem(index, num_context_views, (patchsize_h, patchsize_w))
        except Exception as e:
            print(f"Error: {e}")
            index = np.random.randint(len(self))
            return self.__getitem__((index, num_context_views, patchsize_h))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        
        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return len(self.scene_ids)
