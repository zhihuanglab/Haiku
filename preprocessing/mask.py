# -*- coding: utf-8 -*-
"""
Convolutional Neural Network for tissue FG/BG segmentation

Codes adapted from
https://gitlab.com/enable-medicine/rnd-subgroup/scripts/-/blob/main/ImageAnalysis/tissue_segmentation/cnn_code/model.py

@author: Monee McGrady, Zhenqin Wu
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import LightningModule

TISSUE_MASK_MODEL_WEIGHT_URI = '/home/yancui/tissue_seg_01_20_23.ckpt'


class SegmentationModel(LightningModule):
    def __init__(self,
                 num_classes,
                 lr=1e-4,
                 batch_size=8,
                 pretrained=True):
        super().__init__()
        self.__dict__.update(locals())
        self.criterion = nn.CrossEntropyLoss()
        # Set up backbone
        backbone = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.deeplabv3.DeepLabV3_ResNet50_Weights.DEFAULT)
        self.num_classes = num_classes
        backbone.classifier[4] = torch.nn.Conv2d(256, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.feature_extractor = backbone

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

    def loss(self, logits, labels):
        return self.criterion(logits, labels)

    def accuracy(self, y_pred, y_true):
        correct_pixels = (y_pred == y_true).float().sum()
        total_px = y_pred.shape[0] * y_pred.shape[1]
        return correct_pixels / total_px

    def _sharedstep(self, batch):
        x, y_true = batch
        y_pred = self(x.float())['out']
        return y_pred, y_true.long()

    def training_step(self, batch, batch_idx):
        y_pred, y_true = self._sharedstep(batch)
        loss = self.loss(y_pred, y_true)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, batch, batch_idx):
        y_pred, y_true = self._sharedstep(batch)
        test_loss = self.loss(y_pred, y_true)
        self.log("test_loss", test_loss)
        return (y_true, y_pred)

    def predict_step(self, batch, batch_idx):
        y_true, y_pred = self.test_step(batch, batch_idx)
        y_pred = torch.argmax(y_pred, 1)
        return (y_true, y_pred)

    def validation_step(self, batch, batch_idx):
        y_pred, y_true = self._sharedstep(batch)
        val_loss = self.loss(y_pred, y_true)
        self.log("val_loss", val_loss, on_epoch=True)
        return (y_true, y_pred)


def preprocess_img_for_seg(img):
    """Prepare a single-channel DAPI image for tissue segmentation

    Args:
        img (array-like): single-channel DAPI image

    Returns:
        torch.Tensor: input to CNN model
    """
    img = cv2.resize(img, (256, 256))
    img = img.astype(float)
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    img2 = np.zeros((3, 256, 256))  # channels x w x h
    for i in range(0, 3):
        img2[i, :, :] = img
    img = torch.from_numpy(img2.astype(float))
    return img


def cnn_tissue_mask(img, model):
    """Generate tissue foreground mask for a single-channel DAPI image using
    trained tissue segmentation model

    Args:
        img (array-like): single-channel DAPI image
        model (SegmentationModel): tissue segmentation model

    Returns:
        np.ndarray: 2D binary array of tissue mask
    """
    img_shape = img.shape
    assert len(img_shape) == 3 and img_shape[0] == 1
    dapi_slice = img[0]
    model_input = preprocess_img_for_seg(dapi_slice)
    model_input = model_input.unsqueeze(0).float().to(model.device)

    pred = model(model_input)['out'].cpu()
    pred = torch.argmax(pred, 1).squeeze(0).data.numpy().astype(float)

    pred = cv2.resize(pred, (img_shape[2], img_shape[1]))
    pred[np.where(pred < 1)] = 0
    pred = pred.astype(int)
    assert pred.shape == img.shape[1:]
    return pred


map_location = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = SegmentationModel.load_from_checkpoint(TISSUE_MASK_MODEL_WEIGHT_URI, map_location=map_location, num_classes=2)
model = model.eval()

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import LightningModule

failed_ids = []

from tqdm import tqdm

full_regions_dir = "/data/enable_data/full_regions/"
#new_acq_ids = sorted([d for d in os.listdir(full_regions_dir) if os.path.isdir(os.path.join(full_regions_dir, d))])

# Read new txt id list from mask_failed_ids.log (one id per line)
# Use only regions listed in wrong_regions_with_DAPI.txt
wrong_regions_with_DAPI_path = "/home/yancui/Haiku/wrong_regions_with_DAPI.txt"
if os.path.exists(wrong_regions_with_DAPI_path):
    with open(wrong_regions_with_DAPI_path, "r") as f:
        new_acq_ids = [line.strip() for line in f if line.strip()]
else:
    raise FileNotFoundError(f"{wrong_regions_with_DAPI_path} not found. Please provide this file.")



for acq_id in tqdm(new_acq_ids):
    try:
        target_dir = f"/data/enable_data/full_regions/{acq_id}"
        DAPI_path = os.path.join(target_dir, "DAPI.tif")
        DAPI_img = tifffile.imread(DAPI_path)
        if DAPI_img.shape[1] > 50000:
            print(f"[SKIP] {acq_id}: image too large, skipping normalization.")
        else:
            try:
                DAPI_mask = cnn_tissue_mask(np.expand_dims(DAPI_img, 0), model)
                # Save the DAPI_mask to the same directory with filename "foreground_mask.ome.tif"
                mask_dir = f'/data/enable_data/new_mask/{acq_id}'
                os.makedirs(mask_dir, exist_ok=True)
                out_mask_path = os.path.join(mask_dir, "foreground_mask.ome.tif")
                if not os.path.exists(out_mask_path):
                    tifffile.imwrite(out_mask_path, DAPI_mask.astype(np.uint8))
            except Exception as e:
                failed_ids.append(acq_id)
                print(f"Error processing {acq_id}: {e}")
    except Exception as e:
        failed_ids.append(acq_id)
        print(f"Error processing {acq_id}: {e}")

# After processing, save failed ids to log file
if failed_ids:
    log_file = "mask_failed_ids.log"
    with open(log_file, "w") as f:
        for fid in failed_ids:
            f.write(f"{fid}\n")
    print(f"Failed IDs have been saved to {log_file}")
