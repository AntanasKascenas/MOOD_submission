#  Copyright (C) 2020 Canon Medical Systems Corporation. All rights reserved

from functools import partial

import torch


def get_val_loss(trainer):
    return trainer.state["val_loss"]


def smaller(x1, x2, min_delta=0):
    return x2 - x1 > min_delta


def larger(x1, x2, min_delta=0):
    return x1 - x2 > min_delta


class EarlyStopping:

    def __init__(self, get=get_val_loss, patience=10, better=smaller):

        self.get = get
        self.patience = patience
        self.better = better

        self.best_epoch = None
        self.best = None

    def should_stop(self, trainer) -> bool:

        res = self.get(trainer)
        current_epoch = trainer.state["epoch_no"]

        if not self.best_epoch or self.better(res, self.best):
            self.best = res
            self.best_epoch = current_epoch

        elif current_epoch - self.best_epoch >= self.patience:
            return True

        return False

    def reset(self):
        self.best_epoch = None
        self.best = None


def val_loss_early_stopping(patience: int, min_delta=0):
    return EarlyStopping(get=get_val_loss, patience=patience, better=partial(smaller, min_delta=min_delta))


class ModelSaver:

    def __init__(self, get=lambda trainer: trainer.state["val_loss"], better=smaller, path=None):

        self.get = get
        self.better = better
        self.path = path

        self.best = None

    def save(self, trainer):
        state = trainer.state
        model = trainer.model
        state['best_state_dict'] = {"epoch_no": state["epoch_no"],
                                    "train_it": state["train_it"],
                                    "model_state_dict": model.state_dict(),
                                    "model_class": model.__class__.__name__,
                                    "optimiser_state_dict": trainer.optimiser.state_dict(),
                                    "value": self.best}

        if self.path:
            torch.save(state['best_state_dict'], self.path)


    def check(self, trainer):

        res = self.get(trainer)

        if self.best is None:
            self.best = res
            self.save(trainer)

        else:
            if self.better(res, self.best):
                self.save(trainer)
                self.best = res

    def register(self, callback_dict):
        callback_dict["after_epoch"].append(lambda trainer: self.check(trainer))


def move_to(list, device):
    return [x.to(device) if isinstance(x, torch.Tensor) else x for x in list]