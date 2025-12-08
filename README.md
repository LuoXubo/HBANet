# HBANet: Hybrid Boundary-Aware Attention Network

Official PyTorch implementation of **HBANet**, introduced in “HBANet: A Hybrid Boundary-Aware Attention Network for Infrared and Visible Image Fusion” (CVIU 2024).

Xubo Luo<sup>1</sup>, Jinshuo Zhang<sup>2</sup>, Liping Wang<sup>3</sup>, Dongmei Niu<sup>2</sup>

<sup>1</sup> Shanghai University of Finance and Ecnomics
<sup>2</sup> Jinan University
<sup>3</sup> Jinan Fourth Hospital

<p align="center">
<em>Seamlessly fusing thermal saliency and visible detail via hybrid attention.</em>
</p>

---

## 🔍 Abstract

HBANet unifies infrared (IR) and visible (VIS) imagery through a dual-branch encoder, a Hybrid Boundary-Aware Attention (HBA) module, and a lightweight decoder. The HBA module couples boundary-sensitive spatial attention with cross-domain feature exchange, enabling crisp edge preservation and faithful intensity reconstruction. Training leverages a hybrid fusion loss that balances structure fidelity, brightness consistency, and spatial smoothness.

---

## ✨ Highlights

- **Shared Encoder** – A single convolutional backbone extracts modality-agnostic representations while respecting low-level contrast differences.
- **Hybrid Attention** – BAAU injects VIS-derived boundary priors; CDAU performs bidirectional multi-head attention across IR/VIS streams.
- **Physics-aware Objective** – Structure, intensity, and total variation losses jointly guide fusion quality with default weights $(1.0, 10.0, 0.5)$.
- **Plug-and-Play** – Minimal dependencies, fast inference, and modular design for research or production deployments.

---

## 🧱 Architecture

| Stage | Description |
|-------|-------------|
| Dual-Branch Encoder | Conv–BN–ReLU stack followed by residual blocks (shared weights) produce IR/VIS feature pyramids. |
| Boundary-Aware Attention Unit (BAAU) | Generates a Sobel-based boundary prior from the VIS input to refine spatial saliency. |
| Cross-Domain Attention Unit (CDAU) | Multi-head cross-attention enables global, modality-aware fusion between the two feature streams. |
| Decoder | Residual refinement followed by pointwise projection reconstructs the fused grayscale image. |

Refer to [`details.md`](details.md) for an in-depth breakdown.

---

## ⚙️ Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12 with CUDA support (optional but recommended)
- Additional dependencies listed in `requirements.txt`

```bash
git clone https://github.com/LuoXubo/HBANet.git
cd HBANet
pip install -r requirements.txt
```

---

## 📦 Data Preparation

1. Download paired IR–VIS datasets (e.g., **TNO**, **RoadScene**, **LLVIP**).
2. Align and resize images (default: `256×256`), normalize to `[0, 1]`.
3. Organize directories as required by `data/dataloder.py` (IR and VIS folders with matching filenames).

---

## 🚀 Training

Configure the training option file to enable the hybrid loss (set `G_lossfn_type: hybrid`). A minimal run is launched via:

```bash
python train.py --opt options/train_hbanet.yml
```

Key hyperparameters:

| Parameter | Default |
|-----------|---------|
| Optimizer | Adam (lr = 1e-4, β₁ = 0.9, β₂ = 0.999) |
| Batch size | 8–16 |
| Epochs | 100–200 |
| LR schedule | Cosine decay / StepLR |

---

## 📊 Evaluation

Run inference with a trained checkpoint:

```bash
python test.py \
	--model_path /path/to/checkpoint.pth \
	--dataset_root ./Dataset/testsets \
	--dataset MSRS \
	--ir_dir IR \
	--vis_dir VI \
	--output_dir ./results
```

The script computes fused outputs and stores them under `./results/HBANet_<DATASET>`.

Recommended quantitative metrics: **Entropy (EN)**, **Mutual Information (MI)**, **SSIM**, and **Qabf**. Evaluation utilities can be added in `utils/` or external toolkits.

---

## 📈 Results Snapshot

| Dataset | EN ↑ | MI ↑ | SSIM ↑ | Qabf ↑ |
|---------|------|------|--------|--------|
| MSRS    | TBD  | TBD  | TBD    | TBD    |

*Numbers will be updated once public checkpoints are released.*

---

## 🙏 Acknowledgements

HBANet builds upon insights from:

- [BA-Transformer](https://github.com/jcwang123/BA-Transformer)
- [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion)

---

## 📚 Citation

If our work benefits your research, please cite:

```bibtex
@article{LUO2024104161,
	title   = {HBANet: A hybrid boundary-aware attention network for infrared and visible image fusion},
	journal = {Computer Vision and Image Understanding},
	volume  = {249},
	pages   = {104161},
	year    = {2024},
	doi     = {10.1016/j.cviu.2024.104161},
	author  = {Xubo Luo and Jinshuo Zhang and Liping Wang and Dongmei Niu}
}
```

---

## ✉️ Contact

For questions or collaboration proposals, please open an issue or email **xuboluo@bupt.edu.cn** (replace with the appropriate contact).
