#  Copyright (C) 2020 Canon Medical Systems Corporation. All rights reserved

from pathlib import Path
from math import ceil

import torch
import torch.nn.functional as F
import nibabel as nib

from processing import get_brain_volume, get_abdom_volume, postprocess_brain, postprocess_abdom
from model import noise, UNet

MAX_BATCH = 16

models = Path("/workspace/models")
BRAIN_DENOISING_MODEL = models / "denoising_brain_full_coarse.pt"
BRAIN_SEG_MODEL = models / "seg_brain_full.pt"
ABDOM_DENOISING_MODEL = models / "denoising_abdom_full_coarse.pt"
ABDOM_SEG_MODEL = models / "seg_abdom_full.pt"


def predict(input_folder, target_folder, part="brain", type="pixel"):

    if part == "brain":
        model1 = load_model(BRAIN_DENOISING_MODEL)
        model2 = load_model(BRAIN_SEG_MODEL)
        get_volume = get_brain_volume
        postprocess = postprocess_brain

    elif part == "abdomen":
        model1 = load_model(ABDOM_DENOISING_MODEL)
        model2 = load_model(ABDOM_SEG_MODEL)
        get_volume = get_abdom_volume
        postprocess = postprocess_abdom

    gpu = torch.device("cuda")

    for path in [x.resolve() for x in Path(input_folder).iterdir()]:
        print(f"processing {path}")
        volume, affine = get_volume(path)
        n_batches = int(ceil(volume.shape[-1] / MAX_BATCH))

        output_denoising = torch.zeros_like(volume)
        output_seg = torch.zeros_like(volume)

        # Denoising predictions
        model1 = model1.to(gpu)
        for i in range(n_batches):
            batch = volume[:, :, MAX_BATCH*i: MAX_BATCH*(i+1)]
            batch = batch.permute(2, 0, 1).unsqueeze(dim=1).float().to(gpu)
            with torch.no_grad():
                out = denoising_predict(model1, batch)
            out = out.squeeze(dim=1).permute(1, 2, 0).cpu()
            output_denoising[:, :, MAX_BATCH*i: MAX_BATCH*(i+1)] = out
        model1.cpu()

        # Segmentation predictions
        model2 = model2.to(gpu)
        for i in range(n_batches):
            batch = volume[:, :, MAX_BATCH*i: MAX_BATCH*(i+1)]
            batch = batch.permute(2, 0, 1).unsqueeze(dim=1).float().to(gpu)
            with torch.no_grad():
                out = seg_predict(model2, batch)
            out = out.squeeze(dim=1).permute(1, 2, 0).cpu()
            output_seg[:, :, MAX_BATCH*i: MAX_BATCH*(i+1)] = out
        model2 = model2.cpu()

        output = output_denoising + output_seg
        output /= 2

        if type == "pixel":
            nimg = postprocess(output, affine)
            target_path = Path(target_folder).resolve() / path.parts[-1]
            nib.save(nimg, target_path)
        if type == "sample":
            result = output.sum() / output.numel()
            target_path = Path(target_folder).resolve() / (str(path.parts[-1]) + ".txt")
            with open(target_path, "w") as f:
                f.write(str(result.item()))




def denoising_predict(model, batch, n_it=10):
    mask = batch > 0.01
    errs = torch.zeros_like(batch)
    for i in range(n_it):
        corrupted_input = noise(batch)
        res = (model(corrupted_input) * mask).float()

        eps = 0.1
        std = (F.avg_pool2d((batch - F.avg_pool2d(batch, kernel_size=3, stride=1, padding=1)) ** 2, kernel_size=3, stride=1, padding=1)) ** 0.5 + eps
        err = (batch - res) / std
        errs += err

    errs = errs.abs() / n_it
    errs = F.interpolate(F.interpolate(errs, scale_factor=0.5, mode="bilinear"), scale_factor=2, mode="bilinear")
    return torch.clamp(errs / 1.5, min=0, max=1)



def seg_predict(model, batch):
    mask = batch > 0.05
    out = (model(batch).sigmoid() * mask).float()
    return torch.clamp(out, min=0, max=1)


def load_model(key: str):
    if key == BRAIN_DENOISING_MODEL:
        checkpoint = torch.load(BRAIN_DENOISING_MODEL)
        unet = UNet(in_channels=1, n_classes=1, depth=3, wf=8, padding=True,
                    batch_norm=False, up_mode='upconv', grid=True, bias=True)
        unet.load_state_dict(checkpoint["model_state_dict"])
        return unet

    if key == BRAIN_SEG_MODEL:
        checkpoint = torch.load(BRAIN_SEG_MODEL)
        unet = UNet(in_channels=1, n_classes=1, depth=3, wf=8, padding=True,
                    batch_norm=False, up_mode='upconv', grid=False, bias=True)
        unet.load_state_dict(checkpoint["model_state_dict"])

    if key == ABDOM_DENOISING_MODEL:
        checkpoint = torch.load(ABDOM_DENOISING_MODEL)
        unet = UNet(in_channels=1, n_classes=1, depth=3, wf=8, padding=True,
                    batch_norm=False, up_mode='upconv', grid=True, bias=True)
        unet.load_state_dict(checkpoint["model_state_dict"])

    if key == ABDOM_SEG_MODEL:
        checkpoint = torch.load(ABDOM_SEG_MODEL)
        unet = UNet(in_channels=1, n_classes=1, depth=3, wf=8, padding=True,
                    batch_norm=False, up_mode='upconv', grid=False, bias=True)
        unet.load_state_dict(checkpoint["model_state_dict"])

    return unet

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("-part", type=str, default="brain", help="brain or abdomen", required=True)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    mode = args.mode
    part = args.part

    predict(input_dir, output_dir, part=part, type=mode)
