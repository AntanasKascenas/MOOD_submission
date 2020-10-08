#  Copyright (C) 2020 Canon Medical Systems Corporation. All rights reserved

from pathlib import Path

import nibabel as nib
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F


def get_brain_volume(path: Path):
    nimg = nib.load(path)
    volume = nimg.get_fdata()
    volume = torch.from_numpy(volume).permute(2, 0, 1).unsqueeze(1)
    volume = F.interpolate(volume, scale_factor=0.5, mode="bilinear", align_corners=False).squeeze(1)

    return volume.permute(1, 2, 0), nimg.affine


def get_abdom_volume(path: Path):
    nimg = nib.load(path)
    volume = nimg.get_fdata()
    volume = zoom(volume[:, :, ::2], [0.5, 0.5, 1.])
    volume = torch.from_numpy(volume).permute(2, 0, 1).unsqueeze(1)
    volume = F.interpolate(volume, scale_factor=0.5, mode="bilinear", align_corners=False).squeeze(1)

    return volume.permute(1, 2, 0), nimg.affine


def postprocess_brain(output, affine):
    arr = F.interpolate(output.permute(2, 0, 1).unsqueeze(1), scale_factor=2).squeeze(1).permute(1, 2, 0)

    nimg = nib.Nifti1Image(arr.numpy(), affine=affine)
    return nimg


def postprocess_abdom(output, affine):
    arr = F.interpolate(output.permute(2, 0, 1).unsqueeze(1), scale_factor=2).squeeze(1).permute(1, 2, 0)
    arr = zoom(arr.numpy(), 2)

    nimg = nib.Nifti1Image(arr, affine=affine)
    return nimg
