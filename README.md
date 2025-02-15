# HBANet

Official code of HBANet: A hybrid boundary-aware attention network for infrared and visible image fusion.

## Introduction

This repository contains the official code of the paper "HBANet: A hybrid boundary-aware attention network for infrared and visible image fusion". The code is implemented in PyTorch.

## Installation

1. Clone this repository.

```bash
git clone https://github.com/LuoXubo/HBANet.git
cd HBANet
```

2. Install the required packages.

```bash
pip install -r requirements.txt
```

## Usage

1. Download the dataset and put it in the `data` folder.
2. Train the model (Optional).

```bash
python train.py
```

3. Test the model.

```bash
python test.py
```

## Acknowledgement

The code is based on the following repositories:

- [BA-Transformer](https://github.com/jcwang123/BA-Transformer)
- [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion)

## Citation

If you find this work helpful, please consider citing:

```bibtex
@article{LUO2024104161,
title = {HBANet: A hybrid boundary-aware attention network for infrared and visible image fusion},
journal = {Computer Vision and Image Understanding},
volume = {249},
pages = {104161},
year = {2024},
issn = {1077-3142},
doi = {https://doi.org/10.1016/j.cviu.2024.104161},
url = {https://www.sciencedirect.com/science/article/pii/S107731422400242X},
author = {Xubo Luo and Jinshuo Zhang and Liping Wang and Dongmei Niu}
}
```
