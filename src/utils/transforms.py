import torch
from torchvision import transforms
import numpy as np

def z_score_normalize(img, clip_range=None):
    mean = img.mean(dim=(1, 2), keepdim=True)
    std = img.std(dim=(1, 2), keepdim=True)
    
    normalized_img = (img - mean) / (std + 1e-6) 
    
    if clip_range is not None:
        normalized_img = torch.clamp(normalized_img, clip_range[0], clip_range[1])

    return normalized_img

def min_max_normalize(img):
    min_val = torch.amin(img, dim=(1, 2), keepdim=True)
    max_val = torch.amax(img, dim=(1, 2), keepdim=True)
    
    normalized_img = (img - min_val) / (max_val - min_val + 1e-6)  

    return normalized_img

class MinMaxNormalize:
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-6)
        return normalized_tensor

# class PerChannelSelfStandardization(object):

#     def __init__(self):
#         """
#         Transformation that self-standardizes each channel of the image.
#         """
#         pass

#     def __call__(self, img):
#         if isinstance(img, np.ndarray):
#             mean = img.mean(axis=(-1, -2), keepdims=True)
#             std = img.std(axis=(-1, -2), keepdims=True)
#             std = np.where(std == 0, np.ones_like(std), std)
#             return (img - mean) / std
#         elif isinstance(img, torch.Tensor):
#             mean = img.mean(axis=(-1, -2), keepdims=True)
#             std = img.std(axis=(-1, -2), keepdims=True)
#             std = torch.where(std == 0, torch.ones_like(std), std)
#             return (img - mean) / std
#         else:
#             raise ValueError('Unknown input type')


class DataAugmentationiBOT(object):
    def __init__(self, global_crops_scale=(0.5, 1.), local_crops_scale=(0.2, 0.4), global_crops_number=2, local_crops_number=6):
        flip_and_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ])
    
        self.global_crops_number = global_crops_number
        self.global_transfo = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale),
            flip_and_jitter,
        ])
        
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale),
            flip_and_jitter,
        ])
    
    def __call__(self, img):
        crops = []
        for _ in range(self.global_crops_number):
            crop = self.global_transfo(img)
            crops.append(crop)
        for _ in range(self.local_crops_number):
            crop = self.local_transfo(img)
            crops.append(crop)
        return crops # list of len(total_crops) list[:global_crops_number].shape = B x C x 224 x 224. list[global_crops_number:].shape = B x C x 96 x 96


import torch
import numpy as np

import torch
import torch.nn.functional as F

def build_gaussian_kernel_2d_torch(kernel_size=3, sigma=1.0, device='cpu', dtype=torch.float32):
    """
    Build a 2D Gaussian kernel using torch, returned as (1, 1, k, k) for depthwise conv2d.
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"

    coords = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d /= g1d.sum()
    kernel_2d = torch.outer(g1d, g1d)
    kernel_2d /= kernel_2d.sum()
    return kernel_2d  # shape: (k, k)

def gaussian_blur_2d_torch_per_channel(image_tensor, kernel_2d):
    """
    Applies 2D Gaussian blur independently to each channel of an image (C, H, W).

    Args:
        image_tensor: torch.Tensor of shape (C, H, W)
        kernel_2d: torch.Tensor of shape (k, k)

    Returns:
        blurred: torch.Tensor of shape (C, H, W)
    """
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected image of shape (C, H, W), got {image_tensor.shape}")

    C, H, W = image_tensor.shape
    k = kernel_2d.shape[0]
    pad = k // 2

    # Convert image to (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)

    # Prepare depthwise convolution kernel: (C, 1, k, k)
    kernel = kernel_2d.expand(C, 1, k, k)

    # Apply padding and depthwise convolution
    image_tensor = F.pad(image_tensor, (pad, pad, pad, pad), mode='reflect')
    blurred = F.conv2d(image_tensor, kernel, groups=C)

    return blurred.squeeze(0)

class CustomGaussianBlurTorch:
    """
    Custom Gaussian blur implemented in pure PyTorch for (C, H, W) images,
    applied independently per channel using depthwise conv.
    """
    def __init__(self, kernel_size=3, sigma=1.0, device='cpu', dtype=torch.float32):
        self.kernel_2d = build_gaussian_kernel_2d_torch(kernel_size, sigma, device=device, dtype=dtype)

    def __call__(self, img):
        """
        img: a torch.Tensor of shape (C, H, W)
        returns: blurred img of same shape
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        return gaussian_blur_2d_torch_per_channel(img, self.kernel_2d.to(img.device, img.dtype))

class PerChannelSelfStandardization:
    def __init__(self):
        pass

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            # img: (C, H, W)
            nan_mask = np.isnan(img)                      # shape (C, H, W)
            channel_has_nan = nan_mask.reshape(img.shape[0], -1).any(axis=1)  # (C,)

            mean = img.mean(axis=(1, 2), keepdims=True)
            std = img.std(axis=(1, 2), keepdims=True)
            std[std == 0] = 1.0

            # Standardize
            standardized = (img - mean) / std

            # Zero out channels that have any NaN
            standardized[channel_has_nan[:, None, None]] = 0.0
            return standardized

        elif isinstance(img, torch.Tensor):
            # img: (C, H, W)
            nan_mask = torch.isnan(img)                      # (C, H, W)
            channel_has_nan = nan_mask.flatten(1).any(dim=1) # (C,)

            mean = img.mean(dim=(1, 2), keepdim=True)
            std = img.std(dim=(1, 2), keepdim=True)
            std = torch.where(std == 0, torch.ones_like(std), std)

            standardized = (img - mean) / std

            # Zero out channels that have any NaNs
            standardized = standardized.masked_fill(channel_has_nan[:, None, None], 0.0)
            return standardized

        else:
            raise ValueError('Unknown input type: must be numpy.ndarray or torch.Tensor')

