#  Copyright (C) 2020 Canon Medical Systems Corporation. All rights reserved

from pathlib import Path
import random

import numpy as np
import nibabel as nib
import bcolz
from scipy.ndimage import zoom


def preprocess_brain_train(source_dir: Path, destination_dir: Path):
    # Full Train 700 (out of 800) brain 2d.
    slice_w = 256
    slice_h = 256

    train = list(range(800))
    random.seed(0)
    random.shuffle(train)
    disk_x = bcolz.zeros((0, 1, slice_w, slice_h), rootdir=destination_dir, chunklen=1)

    for i in train[:700]:
        volume = nib.load(source_dir / "{:05}.nii.gz".format(i)).get_fdata()

        for s in range(256):
            x = volume[None, None, :, :, s]
            disk_x.append(x)
            disk_x.flush()


def preprocess_brain_val(source_dir: Path, destination_dir: Path):
    # Full Validation 100 (out of 800) brain 2d.
    slice_w = 256
    slice_h = 256

    train = list(range(800))
    random.seed(0)
    random.shuffle(train)
    disk_x = bcolz.zeros((0, 1, slice_w, slice_h), rootdir=destination_dir, chunklen=1)

    for i in train[700:]:
        volume = nib.load(source_dir / "{:05}.nii.gz".format(i)).get_fdata()

        for s in range(256):
            x = volume[None, None, :, :, s]
            disk_x.append(x)
            disk_x.flush()


def preprocess_abdomen_train(source_dir: Path, destination_dir: Path):
    # Abdom data train 2d, downsample by 2x, 450/500 volumes, all slices
    slice_w = 256  # 512
    slice_h = 256  # 512

    train = list(range(550))
    random.seed(0)
    random.shuffle(train)
    disk_x = bcolz.zeros((0, 1, slice_w, slice_h), rootdir=destination_dir, chunklen=1)

    for i in train[:450]:
        volume = nib.load(source_dir / "{:05}.nii.gz".format(i)).get_fdata()
#        volume = nib.load("../data/abdom/abdom_train/{:05}.nii.gz".format(i)).get_fdata()

        slices = list(range(0, 512, 2))
        for s in slices:
            x = volume[None, None, :, :, s]
            x = zoom(x[0, 0, ...], 0.5, order=2)

            x = x[None, None, ...]
            disk_x.append(x)
            disk_x.flush()


def preprocess_abdomen_val(source_dir: Path, destination_path: Path):
    # Abdom data val 2d, downsample by 2x, 50/500 volumes, all slices
    slice_w = 256  # 512
    slice_h = 256  # 512

    train = list(range(550))
    random.seed(0)
    random.shuffle(train)
    disk_x = bcolz.zeros((0, 1, slice_w, slice_h), rootdir=destination_path, chunklen=1)

    for i in train[450:]:
        volume = nib.load(source_dir / "{:05}.nii.gz".format(i)).get_fdata()
        #volume = nib.load("../data/abdom/abdom_train/{:05}.nii.gz".format(i)).get_fdata()

        slices = list(range(0, 512, 2))
        for s in slices:
            x = volume[None, None, :, :, s]  # .transpose([0, 1, 3, 2])
            x = zoom(x[0, 0, ...], 0.5, order=2)

            x = x[None, None, ...]
            disk_x.append(x)
            disk_x.flush()








