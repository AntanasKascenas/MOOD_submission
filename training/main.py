#  Copyright (C) 2020 Canon Medical Systems Corporation. All rights reserved

from collections import defaultdict
from pathlib import Path
import random
from functools import partial

import torch
import torch.nn.functional as F

from training import simple_train_step, simple_val_step
from model import UNet
from metrics import Loss
from utils import ModelSaver, val_loss_early_stopping
from trainer import Trainer
from data import get_denoising_dataloaders, get_segmentation_dataloaders
from preprocessing import *


def basic_callback_dict(identifier, save="val_loss"):
    callback_dict = defaultdict(list)

    path = Path(__file__).resolve().parent / (identifier+".pt" if identifier else "best_model.pt")
    if save == "val_loss":
        ModelSaver(path=path).register(callback_dict)
    else:
        raise NotImplemented(f"Saving by {save} not implemented.")

    return callback_dict


def adam(params, lr=0.001):
    return torch.optim.Adam(params, lr=lr, amsgrad=True, weight_decay=0.00001)


def denoising(datapath: Path, identifier: str, data="brain", batch_size=8, lr=0.001, depth=3, wf=7):
    device = torch.device("cuda")

    def noise(x):
        std = 0.05
        ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], 16, 16), std=std).to(x.device)
        ns = F.upsample_bilinear(ns, scale_factor=8)

        res = x + ns

        return res

    def loss_f(trainer, batch, batch_results):
        return F.l1_loss(batch_results*(batch[1] > 0.01).float(), batch[1], reduction="mean")

    def forward(trainer, batch):
        if random.random() < 0.3:
            batch[1] = batch[0]
            return trainer.model(batch[0])
        else:
            batch[1] = batch[0]
            batch[0] = noise(batch[0].clone())
            return trainer.model(batch[0])

    model = UNet(in_channels=1, n_classes=1, batch_norm=False, up_mode="upconv", depth=depth, wf=wf, padding=True, grid=True).to(device)

    train_step = partial(simple_train_step, forward=forward, loss_f=loss_f)
    val_step = partial(simple_val_step, forward=forward, loss_f=loss_f)
    optimiser = adam(model.parameters(), lr=lr)
    callback_dict = basic_callback_dict(identifier, save="val_loss")
    Loss(lambda batch_res, batch: loss_f(trainer, batch, batch_res)).register(callback_dict, log=True, tensorboard=True, train=True, val=True)

    trainer = Trainer(model=model,
                      train_dataloader=None,
                      val_dataloader=None,
                      optimiser=optimiser,
                      train_step=train_step,
                      val_step=val_step,
                      callback_dict=callback_dict,
                      device=device,
                      identifier=identifier)

    trainer.noise = noise
    trainer.train_dataloader, trainer.val_dataloader = get_denoising_dataloaders(batch_size, data, datapath)
    trainer.reset_state()
    return trainer


def noise_seg(datapath: Path, identifier=None, lr=0.001, batch_size=8, wf=7, depth=3, data="brain") -> Trainer:
    device = torch.device("cuda")

    def loss_f(trainer, batch, batch_results):
        return (F.binary_cross_entropy_with_logits(batch_results, batch[1].float(), reduction="none") * (batch[0] > 0.05)).mean()

    model = UNet(in_channels=1, n_classes=1, batch_norm=False, up_mode="upconv", depth=depth, wf=wf, padding=True).to(device)

    train_step = partial(simple_train_step, loss_f=loss_f)
    val_step = partial(simple_val_step, loss_f=loss_f)
    optimiser = adam(model.parameters(), lr=lr)
    callback_dict = basic_callback_dict(identifier, save="val_loss")
    Loss(lambda batch_res, batch: loss_f(trainer, batch, batch_res)).register(callback_dict, log=True, tensorboard=True, train=True, val=True)

    trainer = Trainer(model=model,
                      train_dataloader=None,
                      val_dataloader=None,
                      optimiser=optimiser,
                      train_step=train_step,
                      val_step=val_step,
                      callback_dict=callback_dict,
                      device=device,
                      identifier=identifier)

    trainer.train_dataloader, trainer.val_dataloader = get_segmentation_dataloaders(batch_size, data, datapath)
    trainer.reset_state()
    return trainer


if __name__ == "__main__":
    pass

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="path to directory with input data.")
    parser.add_argument("-o", "--output", required=True, type=str, help="path to directory to store preprocessed data.")
    parser.add_argument("--preprocess", required=True, type=bool, default=True, help="whether to run code to do data preprocessing (required once).")
    parser.add_argument("-d", "--data", type=str, default="brain", help="either 'brain' or 'abdomen'.", required=False)
    parser.add_argument("-m", "--method", type=str, default="denoising", help="'denoising' or 'segmentation' selecting which method to train.")

    args = parser.parse_args()

    source_dir = Path(args.input)
    destination_dir = Path(args.output)
    if_preprocess = args.preprocess
    data = args.data
    method = args.method

    if if_preprocess:
        if data == "brain":
            preprocess_brain_train(source_dir, destination_dir / f"{data}_train")
            preprocess_brain_val(source_dir, destination_dir / f"{data}_val")
        elif data == "abdomen":
            preprocess_abdomen_train(source_dir, destination_dir / f"{data}_train")
            preprocess_abdomen_val(source_dir, destination_dir / f"{data}_val")

    if method == "denoising":
        trainer = denoising(identifier=f"denoising_{data}", data=data, batch_size=8, lr=0.001, depth=3, wf=8,
                            datapath=destination_dir)
        trainer.train(epoch_len=480, min_epochs=50, max_epochs=800, early_stopping=val_loss_early_stopping(30))

    elif method == "segmentation":
        trainer = noise_seg(identifier=f"segmentation_{data}", data=data, wf=8, depth=3, lr=0.001, batch_size=8,
                            datapath=destination_dir)
        trainer.train(epoch_len=480, min_epochs=50, max_epochs=800, early_stopping=val_loss_early_stopping(30))



