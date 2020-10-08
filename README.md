This repository stores the training and submission (inference) code for our solution to the [MOOD challenge](https://synapse.org/mood).

# Usage

Use training/main.py to train either of our two methods (denoising and segmentation). The challenge data will need to be downloaded beforehand.

Code in sumbission/ contains the Dockerfile and inference code that can be used to output predictions using saved models. The saved models need to be placed in submission/models and their names hardcoded in submission/predict.py

# Requirements

Models were trained with NVidia GeForce RTX 2080Ti. Models with the default settings use about 8GB of VRAM.

Library requirements are contained in requirements.txt