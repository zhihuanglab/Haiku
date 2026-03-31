import pandas as pd
from einops import rearrange
from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def mask_to_grid(mask, patch_size):
    grid = rearrange(mask, '(h p1) (w p2) -> h w (p1 p2)', p1=patch_size, p2=patch_size)
    return grid

def mask_to_binary(mask, label):
    mask = mask == label
    return mask

def mask_to_binary_grid(mask, label, patch_size):
    mask = mask_to_binary(mask, label)
    mask = mask_to_grid(mask, patch_size).any(dim=-1)
    return mask

def mask_to_predominant(mask, patch_size):
    mask = mask_to_grid(mask, patch_size)
    mask = torch.mode(mask, dim=-1).values
    return mask

def binary_grid_to_mask(grid, patch_size):
    mask = grid.unsqueeze(-1)
    mask = mask.expand(-1, -1, patch_size*patch_size)
    mask = rearrange(mask, 'h w (p1 p2) -> (h p1) (w p2)', p1=patch_size, p2=patch_size)
    return mask

def to_binary_image(mask, label, patch_size):
    mask = mask_to_binary_grid(mask, label, patch_size)
    mask = binary_grid_to_mask(mask, patch_size)
    return mask

def to_composition_vector(mask, num_classes):
    counts = torch.bincount(mask.flatten(), minlength=num_classes)
    return counts / counts.sum()

def mask_to_binary_label(mask, label):
    mask = mask == label
    mask = mask.any(dim=(0,1))
    return mask

def mask_to_predominant_global(mask):
    return mask.flatten().mode().values


class PatchLevelTask(ABC):

    def __init__(self, imc_dataset, task_name):
        self.imc_dataset = imc_dataset
        self.task_name = task_name
    
    @abstractmethod
    def get_full_mask(self, img_name):
        pass

    @abstractmethod
    def mask_to_label_grid(self, mask, patch_size):
        pass

class PatchLevelBinaryTask(PatchLevelTask):

    def __init__(self, imc_dataset, task_name, cell_type_id, column_name="cell_category"):
        super().__init__(imc_dataset, task_name)
        self.column_name = column_name
        self.cell_type_id = cell_type_id

    def get_full_mask(self, img_name):
        mask = self.imc_dataset.load_cell_type_mask(img_name, column_name=self.column_name)
        return mask

    def mask_to_label_grid(self, mask, patch_size):
        return mask_to_binary_grid(mask, self.cell_type_id, patch_size)

class PatchLevelInteractionTask(PatchLevelTask):

    def __init__(self, imc_dataset, task_name, cell_type_id_source, cell_type_id_target, column_name="cell_category", kernel_size=3):
        super().__init__(imc_dataset, task_name)
        self.column_name = column_name
        self.cell_type_id_source = cell_type_id_source
        self.cell_type_id_target = cell_type_id_target
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
    
    def get_full_mask(self, img_name):
        mask = self.imc_dataset.load_cell_type_mask(img_name, column_name=self.column_name)
        return mask
    
    def mask_to_label_grid(self, mask, patch_size):
        source_mask = mask_to_binary_grid(mask, self.cell_type_id_source, patch_size)
        target_mask = mask_to_binary_grid(mask, self.cell_type_id_target, patch_size).to(mask.dtype).unsqueeze(0)
        kernel = torch.ones(1,1,self.kernel_size,self.kernel_size, dtype=mask.dtype)
        target_mask_convolved = torch.nn.functional.conv2d(target_mask, kernel, padding=self.padding, dilation=1, groups=1).squeeze(0)
        target_mask_convolved = target_mask_convolved > 0
        return source_mask & target_mask_convolved


class PatchLevelMulticlassTask(PatchLevelTask):

    def __init__(self, imc_dataset, task_name, column_name):
        super().__init__(imc_dataset, task_name)
        self.column_name = column_name

    def get_full_mask(self, img_name):
        mask = self.imc_dataset.load_cell_type_mask(img_name, column_name=self.column_name)
        return mask

    def mask_to_label_grid(self, mask, patch_size):
        return mask_to_predominant(mask, patch_size)

class CropLevelTask(ABC):

    def __init__(self, imc_dataset, task_name, requires_mlp=False):
        self.imc_dataset = imc_dataset
        self.task_name = task_name
        self.requires_mlp = requires_mlp
    
    @abstractmethod
    def get_full_mask(self, img_name):
        pass

    @abstractmethod
    def mask_to_label(self, mask):
        pass

class CropLevelBinaryTask(CropLevelTask):

    def __init__(self, imc_dataset, task_name, cell_type_id, column_name, requires_mlp=False):
        super().__init__(imc_dataset, task_name, requires_mlp=requires_mlp)
        self.column_name = column_name
        self.cell_type_id = cell_type_id

    def get_full_mask(self, img_name):
        mask = self.imc_dataset.load_cell_type_mask(img_name, column_name=self.column_name)
        return mask
    
    def mask_to_label(self, mask):
        return mask_to_binary_label(mask, self.cell_type_id)

class CropLevelMulticlassTask(CropLevelTask):

    def __init__(self, imc_dataset, task_name, column_name, requires_mlp=False):
        super().__init__(imc_dataset, task_name, requires_mlp=requires_mlp)
        self.column_name = column_name

    def get_full_mask(self, img_name):
        mask = self.imc_dataset.load_cell_type_mask(img_name, column_name=self.column_name)
        return mask
    
    def mask_to_label(self, mask):
        return mask_to_predominant_global(mask)

class CropLevelRegressionTask(CropLevelTask):

    def __init__(self, imc_dataset, task_name, num_targets, column_name, requires_mlp=False):
        super().__init__(imc_dataset, task_name, requires_mlp=requires_mlp)
        self.column_name = column_name
        self.num_targets = num_targets

    def get_full_mask(self, img_name):
        mask = self.imc_dataset.load_cell_type_mask(img_name, column_name=self.column_name)
        return mask
    
    def mask_to_label(self, mask):
        return to_composition_vector(mask, num_classes=self.num_targets)

class ImageLevelTask(ABC):

    def __init__(self, imc_dataset, task_name, final_eval=False):
        self.imc_dataset = imc_dataset
        self.task_name = task_name
        self.final_eval = final_eval
    
    @abstractmethod
    def get_label(self, img_name):
        pass

class ImageLevelMulticlassTask(ImageLevelTask):

    def __init__(self, imc_dataset, task_name, column_name, final_eval=False):
        super().__init__(imc_dataset, task_name, final_eval=final_eval)
        self.column_name = column_name
    
    def get_label(self, img_name):
        df = self.imc_dataset.get_image_index()
        return str(df[df["image_name"] == img_name][self.column_name].iloc[0])
    
class ImageLevelRegressionTask(ImageLevelTask):

    def __init__(self, imc_dataset, task_name, column_name, final_eval=False):
        super().__init__(imc_dataset, task_name, final_eval=final_eval)
        self.column_name = column_name

    def get_label(self, img_name):
        df = self.imc_dataset.get_image_index()
        return df[df["image_name"] == img_name][self.column_name].iloc[0]
    
class CropLevelBinaryStructureTask(CropLevelTask):

    def __init__(self, imc_dataset, task_name, column_name, requires_mlp=False):
        super().__init__(imc_dataset, task_name, requires_mlp=requires_mlp)
        self.column_name = column_name

    def get_full_mask(self, img_name):
        mask = self.imc_dataset.load_structure_mask(img_name, column_name=self.column_name)
        return mask

    def mask_to_label(self, mask):
        return mask_to_binary_label(mask, 1)