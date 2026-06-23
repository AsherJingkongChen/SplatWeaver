# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as tvf
from PIL.ImageOps import exif_transpose
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(folder_or_list, size, square_ok=False, verbose=True, rotate_clockwise_90=False, crop_to_landscape=False):
    """open and convert all images in a list or folder to proper input format for DUSt3R"""
    if isinstance(folder_or_list, str):
        if verbose:
            print(f">> Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f">> Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list

    else:
        raise ValueError(f"bad {folder_or_list=} ({type(folder_or_list)})")

    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert("RGB")
        if rotate_clockwise_90:
            img = img.rotate(-90, expand=True)
        if crop_to_landscape:
            # Crop to a landscape aspect ratio (e.g., 16:9)
            desired_aspect_ratio = 4 / 3
            width, height = img.size
            current_aspect_ratio = width / height

            if current_aspect_ratio > desired_aspect_ratio:
                # Wider than landscape: crop width
                new_width = int(height * desired_aspect_ratio)
                left = (width - new_width) // 2
                right = left + new_width
                top = 0
                bottom = height
            else:
                # Taller than landscape: crop height
                new_height = int(width / desired_aspect_ratio)
                top = (height - new_height) // 2
                bottom = top + new_height
                left = 0
                right = width

            img = img.crop((left, top, right, bottom))

        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not (square_ok) and W == H:
                halfh = 3 * halfw / 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        W2, H2 = img.size
        if verbose:
            print(f" - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")
        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )

    assert imgs, "no images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")
    return imgs

def process_image(img_path):
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        # Convert RGBA to RGB by removing alpha channel
        img = img.convert('RGB')
    # Resize to maintain aspect ratio and then center crop to 448x448
    width, height = img.size
    if width > height:
        new_height = 448
        new_width = int(width * (new_height / height))
    else:
        new_width = 448
        new_height = int(height * (new_width / width))
    img = img.resize((new_width, new_height))
    
    # Center crop
    left = (new_width - 448) // 2
    top = (new_height - 448) // 2
    right = left + 448
    bottom = top + 448
    img = img.crop((left, top, right, bottom))
    img_tensor = torchvision.transforms.ToTensor()(img) * 2.0 - 1.0 # [-1, 1]
    return img_tensor


import torchvision.transforms as transforms


def adjust_short_side_to_multiple_of_14(size):
    def closest_divisible_by_14(n):
        return round(n / 14) * 14

    return closest_divisible_by_14(size)


def process_image_448(img_path):
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        # Convert RGBA to RGB by removing alpha channel
        img = img.convert('RGB')
    width, height = img.size
    if width > height:
        # Long side is width, so resize based on width
        new_width = 448
        ratio = new_width / width
        new_height = int(height * ratio)
    else:
        # Long side is height, so resize based on height
        new_height = 448
        ratio = new_height / height
        new_width = int(width * ratio)

    # Adjust the short side to be divisible by 14
    if new_width > new_height:
        new_height = adjust_short_side_to_multiple_of_14(new_height)
    else:
        new_width = adjust_short_side_to_multiple_of_14(new_width)

    img = img.resize((new_width, new_height))

    img_tensor = transforms.ToTensor()(img) * 2.0 - 1.0  # [-1, 1]
    return img_tensor


def process_image_252(img_path, target_size=(448, 252)):

    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    w_in, h_in = img.size
    w_out, h_out = target_size  # 448, 252

    scale_factor = max(w_out / w_in, h_out / h_in)
    w_scaled = round(w_in * scale_factor)
    h_scaled = round(h_in * scale_factor)

    img = img.resize((w_scaled, h_scaled), Image.LANCZOS)

    left = (w_scaled - w_out) // 2
    top = (h_scaled - h_out) // 2
    right = left + w_out
    bottom = top + h_out

    img = img.crop((left, top, right, bottom))

    img_tensor = transforms.ToTensor()(img) * 2.0 - 1.0

    return img_tensor



def visualize_expert_selection(expert_onehot, save_path=None):
    """
    Visualize a (4, H, W) one-hot tensor as a color map.

    Args:
        expert_onehot: torch.Tensor or np.ndarray of shape (4, H, W), one-hot encoded.
        save_path: Optional[str], path to save the image (e.g., 'expert_map.png')
    """
    if isinstance(expert_onehot, torch.Tensor):
        expert_onehot = expert_onehot.cpu().numpy()

    assert expert_onehot.shape[0] == 4, "First dim must be 4 (for 4 experts)"

    colors = np.array([
        [0, 0, 255],  # Expert 0: Blue
        [0, 255, 0],  # Expert 1: Green
        [255, 255, 0],  # Expert 2: Yellow
        [255, 0, 0],  # Expert 3: Red
    ], dtype=np.uint8)

    expert_index = np.argmax(expert_onehot, axis=0)  # Shape: (H, W)

    H, W = expert_index.shape
    rgb_image = colors[expert_index]  # Shape: (H, W, 3)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.axis('off')

    # Create a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Expert 0'),
        Patch(facecolor='green', label='Expert 1'),
        Patch(facecolor='yellow', label='Expert 2'),
        Patch(facecolor='red', label='Expert 3'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.0))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()