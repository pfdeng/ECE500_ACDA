"""Smoke tests for the CIFAR/ACDA models and training step.

Run directly (no pytest needed):
    .venv/bin/python tests/test_smoke.py
Or with pytest:
    .venv/bin/python -m pytest tests/
"""
import os
import sys

import torch

# Make the CIFAR/ package importable (models/, ACDA/ live there).
CIFAR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CIFAR"))
sys.path.insert(0, CIFAR_DIR)

from models import LeNet_ACDA, LeNet, ResNet18, ACDA_modified  # noqa: E402


def _assert_forward(model, n=4):
    x = torch.randn(n, 3, 32, 32)
    y = model(x)
    assert y.shape == (n, 10), y.shape


def test_lenet_acda_forward():
    _assert_forward(LeNet_ACDA(10))


def test_lenet_forward():
    _assert_forward(LeNet(10))


def test_resnet18_forward():
    _assert_forward(ResNet18())


def test_acda_alias():
    # ACDA_modified was the historical name; it must remain a valid alias.
    assert ACDA_modified is LeNet_ACDA


def test_one_train_step():
    torch.manual_seed(0)
    model = LeNet_ACDA(10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    x = torch.randn(8, 3, 32, 32)
    target = torch.randint(0, 10, (8,))

    losses = []
    for _ in range(3):
        opt.zero_grad()
        loss = crit(model(x), target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(v == v for v in losses), "loss became NaN"
    assert losses[-1] <= losses[0] + 1e-3, f"loss increased: {losses}"


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
            passed += 1
        except Exception:
            print("FAIL", t.__name__)
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
