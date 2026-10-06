"""Optimizers compatible with the Docker-pinned PyTorch 2.1 stack.

PyTorch 2.2 added ``decoupled_weight_decay`` to ``RAdam``. The images pin 2.1,
where that kwarg raises ``TypeError``. This factory keeps AdamW-style decay:
parameters are scaled by ``(1 - lr * weight_decay)`` before the adaptive step,
and the inner RAdam sees ``weight_decay=0`` so the penalty is not applied twice.
"""
from __future__ import annotations

import inspect
from typing import Iterable

import torch


class DecoupledRAdam(torch.optim.RAdam):
    """RAdam with AdamW-style weight decay, for PyTorch builds that lack the flag."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, **kwargs):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0, **kwargs)
        for group in self.param_groups:
            group["decoupled_weight_decay"] = weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            decay = group.get("decoupled_weight_decay", 0.0)
            if not decay:
                continue
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.mul_(1.0 - lr * decay)
        return super().step(closure)


def build_radam(params: Iterable, *, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Construct RAdam with decoupled weight decay on PyTorch 2.1 and 2.2+."""
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if "decoupled_weight_decay" in inspect.signature(torch.optim.RAdam).parameters:
        kwargs["decoupled_weight_decay"] = True
        return torch.optim.RAdam(params, **kwargs)
    return DecoupledRAdam(params, lr=lr, weight_decay=weight_decay)
