"""RAdam construction must succeed on the Docker-pinned PyTorch API."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR = REPO_ROOT / "nvidia-a2sb-original-repo"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from training_optimizer import DecoupledRAdam, build_radam


def test_build_radam_constructs_without_unexpected_kwargs():
    param = torch.nn.Parameter(torch.ones(4))
    optimizer = build_radam([param], lr=1e-4, weight_decay=0.01)
    assert isinstance(optimizer, torch.optim.Optimizer)
    param.grad = torch.zeros_like(param)
    optimizer.step()


def test_decoupled_radam_shrinks_parameters_before_the_adaptive_step():
    param = torch.nn.Parameter(torch.ones(4))
    optimizer = DecoupledRAdam([param], lr=0.1, weight_decay=0.5)
    param.grad = torch.zeros_like(param)
    optimizer.step()
    # Zero grad → adaptive update is a no-op; only decoupled decay remains.
    # 1 - lr * wd = 1 - 0.05 = 0.95
    assert torch.allclose(param, torch.full_like(param, 0.95))


def test_build_radam_matches_lightning_module_call_shape():
    """configure_optimizers passes model parameters, lr, and weight_decay only."""
    model = torch.nn.Linear(3, 2)
    optimizer = build_radam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


class _Torch21RAdam(torch.optim.Optimizer):
    """Stand-in for torch 2.1 RAdam: no decoupled_weight_decay argument."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, foreach=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        return None


def test_build_radam_constructs_when_decoupled_kwarg_is_missing(monkeypatch):
    """The Docker image pins torch 2.1; RAdam.__init__ rejects that kwarg."""
    monkeypatch.setattr(torch.optim, "RAdam", _Torch21RAdam)
    param = torch.nn.Parameter(torch.ones(4))
    optimizer = build_radam([param], lr=1e-4, weight_decay=0.01)
    param.grad = torch.ones_like(param)
    optimizer.step()


def test_in_step_clipping_mutates_leftover_gradients_before_backward():
    """Reproduce why clip_grad_norm_ must not run inside training_step.

    With accumulate_grad_batches=4 the leftover grad from earlier micro-batches
    is still on the parameter. Clipping it before this step's backward shrinks
    that remainder (2 → 0.5 here) and Lightning clips again at the update.
    """
    param = torch.nn.Parameter(torch.ones(1))
    param.grad = torch.tensor([2.0])
    torch.nn.utils.clip_grad_norm_([param], max_norm=0.5)
    assert float(param.grad) == pytest.approx(0.5)


def test_training_step_does_not_clip_gradients():
    for name in ("A2SB_lightning_module.py", "A2SB_lightning_module_api.py"):
        source = (VENDOR / name).read_text(encoding="utf-8")
        body = source.split("def training_step", 1)[1].split("\n    def ", 1)[0]
        assert "clip_grad_norm_" not in body
        assert "def on_before_optimizer_step" in source
