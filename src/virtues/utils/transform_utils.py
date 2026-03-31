import numpy as np
import random
import torch
from torchvision.transforms import v2
import math
from einops import rearrange

class DropChannels(object):

    def __init__(self, p=0.5, fraction_range=[0.5,0.5]):
        """
        An image transformation for multiplex images that randomly drops a fraction of the channels.
        Args:
            p (float): the probability of applying the transformation
            fraction_range (tuple): the range of the fraction of channels to keep
        """
        self.p = p
        self.fraction_range = fraction_range

    def __call__(self, *args):

        if np.random.rand() < self.p or self.p == 1.0:
            
            lengths = set([len(arg) for arg in args])
            assert len(lengths) == 1, 'All inputs must have the same number of channels'

            num_channels = len(args[0])

            fraction = random.uniform(*self.fraction_range)
            num_channels_to_keep = np.ceil(num_channels * fraction).astype(int)

            indices = np.random.choice(num_channels, num_channels_to_keep, replace=False)
            return [arg[indices] for arg in args]
        
        else:
            return args
        
class CustomGaussianBlur(object):

    def __init__(self, kernel_size, sigma):
        """
        A Gaussian blur transformation customized to work on both numpy arrays and PyTorch tensors.
        """
        self.transform = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, img):
        """
        Applies transformation to the image.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
            return self.transform(img).numpy()
        else:
            return self.transform(img)

class RandomRotation90(object):

    def __init__(self, p=0.5):
        """
        Transformation that rotates at random by 90 degrees and flips by axis.
        """
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return img.transpose(1, 2)
        return img
    
class CropToPatchSize(object):

    def __init__(self, patch_size):
        """
        Transformation that crops the image to a size that is a multiple of the patch size.
        """
        self.patch_size = patch_size

    def __call__(self, img):
        num_patches_x  = img.shape[-2] // self.patch_size
        num_patches_y = img.shape[-1] // self.patch_size
        return img[..., :num_patches_x * self.patch_size, :num_patches_y * self.patch_size]

class GridReshape(object):

    def __init__(self, patch_size):
        """
        Transformation that reshapes the image into a grid of tokens.
        """
        self.patch_size = patch_size

    def __call__(self, img):
        assert img.shape[-2] % self.patch_size == 0 and img.shape[-1] % self.patch_size == 0, 'Image dimensions must be divisible by patch size'
        return rearrange(img, 'c (h p1) (w p2) -> c h w (p1 p2)', p1=self.patch_size, p2=self.patch_size)

class PerChannelRescale(object):

    def __init__(self):
        """
        Transformation that rescales each channel of the image to the range [0, 1].
        """
        pass

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            m = img.max(axis=(-1, -2), keepdims=True)
            m = np.where(m == 0, np.ones_like(m), m)
            img = img / m
            return img
        elif isinstance(img, torch.Tensor):
            m = img.max(dim=-1)[0].max(dim=-1)[0][:,None,None]
            m = torch.where(m == 0, torch.ones_like(m), m)
            img = img / m
            return img
        else:
            raise ValueError('Unknown input type')

class PerChannelSelfStandardization(object):

    def __init__(self):
        """
        Transformation that self-standardizes each channel of the image.
        """
        pass

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            mean = img.mean(axis=(-1, -2), keepdims=True)
            std = img.std(axis=(-1, -2), keepdims=True)
            std = np.where(std == 0, np.ones_like(std), std)
            return (img - mean) / std
        elif isinstance(img, torch.Tensor):
            mean = img.mean(axis=(-1, -2), keepdims=True)
            std = img.std(axis=(-1, -2), keepdims=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            return (img - mean) / std
        else:
            raise ValueError('Unknown input type')
    
class PerChannelSelfStandardizationNoCentering(object):

    def __init__(self):
        """
        Transformation that self-standardizes each channel of the image without centering. Recommended in https://arxiv.org/pdf/2301.05768 for microscopy images.
        """
        pass

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            std = img.std(axis=(-1, -2), keepdims=True)
            std = np.where(std == 0, np.ones_like(std), std)
            return img / std
        elif isinstance(img, torch.Tensor):        
            std = img.std(axis=(-1, -2), keepdims=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            return img / std
        else:
            raise ValueError('Unknown input type')
        
class PerChannelUnitSecondMomentum(object):

    def __init__(self):
        """
        Transformation that divides each channel of the image by its second momentum.
        """
        pass

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            sec_momentum = np.sqrt((img**2).mean(axis=(-1, -2), keepdims=True))
            sec_momentum = np.where(sec_momentum == 0, np.ones_like(sec_momentum), sec_momentum)
            return img / sec_momentum
        elif isinstance(img, torch.Tensor):
            sec_momentum = img.pow(2).mean(axis=(-1, -2), keepdims=True).sqrt()
            sec_momentum = torch.where(sec_momentum == 0, torch.ones_like(sec_momentum), sec_momentum)
            return img / sec_momentum
        else:
            raise ValueError('Unknown input type')

def get_normalization_transform(name):
    """
    Gets specified normalization transform.
    Args:
        name: the name of the normalization transform
    """
    if name == 'unit_rescale':
        return PerChannelRescale()
    elif name == 'self_std':
        return PerChannelSelfStandardization()
    elif name == 'self_std_no_center':
        return PerChannelSelfStandardizationNoCentering()
    elif name == 'unit_second_momentum':
        return PerChannelUnitSecondMomentum()
    elif name == "none":
        return lambda x: x
    else:
        raise ValueError(f'Unknown normalization transform {name}')


class MultiImageRandomCrop(object):
    """
    Crops the same view out of a list of images of equal size. Useful to crop image and masks simultaneously.
    """
    def __init__(self, size, return_coordinates=False):
        self.size = size
        self.return_coordinates = return_coordinates

    def __call__(self, *images):
        target_h = self.size[0]
        target_w = self.size[1]
        output = []

        h, w = images[0].shape[-2], images[0].shape[-1]

        r = random.randint(0, h - target_h)
        c = random.randint(0, w - target_w)
        for i in range(len(images)):
            img = images[i]
            img = img[..., r:r+target_h, c:c+target_w]
            output.append(img)

        if self.return_coordinates:
            return output, (r, c)
        else:
            return output

def sample_mask_area(H, W, mask_ratio):
    """
    Samples number of masked patches uniformly from specified range.
    Args:
        H: the height of the grid
        W: the width of the grid
        mask_ratio: tuple of the range of the mask ratio
    """
    mask_ratio_upper = mask_ratio[1]
    mask_ratio_lower = mask_ratio[0]
    rnd_mask_ratio = random.uniform(mask_ratio_lower, mask_ratio_upper)
    mask_volume = math.ceil(H * W * rnd_mask_ratio)
    return mask_volume

def generate_mask(C, H, W, mask_ratio, mask_strategy):
    """
    Randomly generates a mask for the image.
    Args:
        C: the number of channels
        H: the height of the grid
        W: the width of the grid
        mask_ratio: tuple of the range of the mask ratio
        mask_strategy: the masking strategy, either 'uniform_independent' (corresponds to independent masking in paper) or 'uniform_coupled' (corresponds to niche masking in paper). 
    """
    if mask_strategy == 'uniform_independent':
        masks = []
        for _ in range(C):
            mask_area = sample_mask_area(H, W, mask_ratio)
            mask = torch.zeros(H*W, dtype=bool)
            mask[:mask_area] = True
            mask = mask[torch.randperm(H*W)].reshape(H, W)
            masks.append(mask)
        mask = torch.stack(masks, dim=0)

    elif mask_strategy == "uniform_coupled":
        mask_area = sample_mask_area(H, W, mask_ratio)
        total_area = H * W
        mask =  torch.zeros(H*W, dtype=bool)
        mask[:mask_area] = True
        mask = mask[torch.randperm(total_area)].reshape(H, W)
        mask = mask.unsqueeze(0).expand(C, H, W)

    else:
        raise ValueError(f"Unknown mask strategy {mask_strategy}")
    return mask   
    