#  Copyright (C) 2020 Canon Medical Systems Corporation. All rights reserved

import torch


def simple_forward(trainer, batch):
    return trainer.model(batch[0])


def simple_loss(trainer, batch, res):
    loss = trainer.additional_params["loss"]
    return loss(res, batch[-1])


def simple_train_step(trainer, forward=simple_forward, loss_f=simple_loss):
    state = trainer.state

    trainer.optimiser.zero_grad()

    batch = state["train_batch"]
    y_pred = forward(trainer, batch)
    state["train_batch_result"] = y_pred.detach()
    loss = loss_f(trainer, batch, y_pred)

    with torch.no_grad():
        if torch.isnan(loss).sum() >= 1:
            raise ValueError("nan loss encountered, stopping...")

    if loss.requires_grad:
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(trainer.model.parameters(), max_norm=0.5)
        trainer.optimiser.step()


def simple_val_step(trainer, forward=simple_forward, loss_f=simple_loss):

    state = trainer.state
    with torch.no_grad():
        y_pred = forward(trainer, state["val_batch"])
    state["val_batch_result"] = y_pred

