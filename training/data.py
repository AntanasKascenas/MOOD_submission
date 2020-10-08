#  Copyright (C) 2020 Canon Medical Systems Corporation. All rights reserved

from pathlib import Path
import random
from functools import partial

import bcolz
import torch
import numpy as np
from torch.nn import functional as F

from anomalies import get_all_anomalies

DATAPATH = Path(__file__).parent.parent / "data"


class ArrayDataset(torch.utils.data.Dataset):
    """
    Returns processed bcolz items.
    x_array: bcolz array
    y_array: bcolz array
    process: function (x_item, y_item -> dataset return type)
    index_map: dictionary mapping the torch dataset indices to the bcolz array indices.
    """

    def __init__(self, x_array, y_array, process):
        self.x_array = x_array
        self.y_array = y_array
        self.process = process

    def __getitem__(self, idx):
        return self.process(self.x_array[idx], self.y_array[idx])

    def __len__(self):
        return len(self.x_array)


def insert_anomaly(x, y, anomalies: list, scale_factor=0.5):
    anomaly = random.choice(anomalies)
    if (x[0] > 0.05).sum() < 10:
        # Don't bother inserting an anomaly if slice is mostly empty.
        x = x[0]
        y = np.zeros_like(x)
    else:
        x, y = anomaly(x[0], seed=None)

    x = torch.from_numpy(x).view(1, 1, x.shape[-2], x.shape[-1]).float()
    x = F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)

    y = torch.from_numpy(y).view(1, 1, y.shape[-2], y.shape[-1]).float()
    y = F.interpolate(y, scale_factor=scale_factor, mode="bilinear", align_corners=False)

    return x[0], y[0]


def reconstruction(x, y, scale_factor=0.5):
    x = torch.from_numpy(x).view(1, 1, x.shape[-2], x.shape[-1]).float()
    x = F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)
    return x[0], x[0]


def get_anomalyseg_data(datapath: Path, scale_factor=0.5):

    arr = bcolz.open(datapath)
    all_anomalies = get_all_anomalies()
    result = ArrayDataset(arr, arr, process=partial(insert_anomaly, anomalies=all_anomalies, scale_factor=scale_factor))
    return result


def get_anomaly_data(datapath: Path, scale_factor=0.5):

    arr = bcolz.open(datapath)
    return ArrayDataset(arr, arr, process=partial(reconstruction, scale_factor=scale_factor))


def get_denoising_dataloaders(datapath: Path, batch_size=8, data="brain"):
    path = datapath

    val_dataset = get_anomaly_data(scale_factor=0.5, datapath=path/f"{data}_val")
    train_dataset = get_anomaly_data(scale_factor=0.5, datapath=path/f"{data}_train")

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)

    return train_dataloader, val_dataloader


def get_segmentation_dataloaders(datapath: Path, batch_size=8, data="brain"):
    path = datapath

    val_dataset = get_anomalyseg_data(scale_factor=0.5, datapath=path/f"{data}_val")
    train_dataset = get_anomalyseg_data(scale_factor=0.5, datapath=path/f"{data}_train")

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)

    return train_dataloader, val_dataloader
