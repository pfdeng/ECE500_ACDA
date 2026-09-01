# ECE500ML Final Project — ACDA

Image classification on **CIFAR-10 / CIFAR-100** in PyTorch, comparing standard
CNN baselines against an **Adaptive Convolution with Dynamic Atoms (ACDA)**
variant of LeNet. The ACDA layer generates per-pixel convolution filters from a
small filter-generating network combined with learned spatial position atoms
(see [`CIFAR/ACDA/Conv_re.py`](CIFAR/ACDA/Conv_re.py)).

Inspired by [Zijin Luo's pytorch-cifar10](https://github.com/soapisnotfat/pytorch-cifar10).

## Layout

```
CIFAR/
  main.py            # training / evaluation entry point (CIFAR-10)
  misc.py            # progress-bar utility
  ACDA/Conv_re.py    # the ACDA adaptive-convolution layer
  models/            # LeNet, LeNet_ACDA, ResNet, Ad_ResNet, AlexNet
  *.ipynb            # exploratory notebooks
tests/test_smoke.py  # model + train-step smoke tests
requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

`main.py` trains `LeNet_ACDA` on CIFAR-10 by default (switch the model in
`Solver.load_model`). CIFAR-10 is downloaded automatically on first run.

```bash
cd CIFAR
python main.py --epoch 50 --lr 0.001 --trainBatchSize 100
```

Device is auto-selected (`cuda` > `mps` > `cpu`); force one with `--device`:

```bash
python main.py --device mps --epoch 50      # Apple Silicon GPU
```

Quick smoke run (a few batches, verifies the pipeline end to end):

```bash
python main.py --epoch 1 --max-train-batches 5 --max-test-batches 5
```

The best model is saved to `model.pth` at the end of training.

## Test

```bash
python tests/test_smoke.py          # no extra deps
# or, if pytest is installed:
python -m pytest tests/
```

## Datasets

CIFAR-10 and CIFAR-100 are downloaded directly by torchvision into `CIFAR/data/`
(git-ignored). For CIFAR-100, set the classifier `out_size` to 100 when
constructing the model.
