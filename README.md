# Spatio-Temporal U-TAE for Next-Day Wildfire Spread Prediction

An end-to-end deep learning framework for next-day active wildfire prediction using multi-modal satellite imagery and meteorological time-series. The model adapts the **U-TAE (U-Net with Lightweight Temporal Attention Encoder)** architecture to ingest 5-day sequences across 17 spatial-temporal channels and generate high-resolution active fire probability maps on day $t+1$.

---

## 1. Project Overview

Wildfires exhibit complex spatial dynamics driven by temporal trends in fuel moisture, atmospheric conditions, and topography. This repository implements:
- A spatio-temporal neural network based on **U-TAE** that combines multi-scale spatial convolutional feature extraction with multi-head temporal attention.
- A 17-channel multi-modal representation covering weather conditions, vegetation indices, fire intensity, historical active fire detections, and observational quality masks.
- Hybrid **Combined Loss (Focal Loss + Soft Dice Loss)** specifically designed to mitigate the extreme sparsity of active fire pixels.
- Direct benchmark comparison against the standard **Persistence Baseline** on the held-out **2023 evaluation test set**.
- An interactive web application built with **Gradio**, fully compatible with **Hugging Face Spaces** and **ZeroGPU**.

---

## 2. Model Architecture

The architecture builds upon the **U-TAE** framework originally introduced for satellite time-series segmentation:

```
Input: (B, T=5, C=17, H=128, W=128)
  │
  ▼
[Spatial CNN Encoder] ────► 4 stages: [64, 64, 64, 128], GroupNorm, ReLU
  │
  ▼ (Feature maps at t=1..5, lowest spatial resolution: 16x16)
[L-TAE Temporal Bottleneck] ─► 16-head Multi-Head Temporal Self-Attention (d_k=4)
  │                            Sinusoidal temporal positional encodings
  │                            Generates temporal attention map: (16, B, T, 16, 16)
  ▼
[Temporal Aggregator] ────► Attention-weighted temporal fusion for skip connections
  │
  ▼
[Spatial CNN Decoder] ────► 4 stages: [128, 64, 32, 32] via transposed convs & skip fusion
  │
  ▼
[Output Head] ────────────► Conv2d(32 -> 1, 1x1)
  │
  ▼
Output: (B, 1, 128, 128) ─── Logits / Sigmoid Probabilities for Fire[t+1]
```

### Key Components
1. **Multi-Scale Spatial Encoder:** Uses 4 strided downsampling convolution blocks (`str_conv_k=4, str_conv_s=2, str_conv_p=1`) shared across all $T=5$ time steps.
2. **L-TAE (Lightweight Temporal Attention Encoder):** Computes pixel-wise multi-head self-attention across the 5 input days. Unlike recurrent architectures (RNN/LSTM), L-TAE captures non-local temporal correlations with fewer parameters and no vanishing gradients.
3. **Temporal Aggregator (`att_group`):** Resamples the L-TAE attention masks to match encoder skip resolutions and computes attention-weighted temporal sums to feed into decoder skip connections.
4. **Spatial Decoder:** Progressively restores spatial resolution to $128 \times 128$, concatenating attention-aggregated skip features at each scale.

---

## 3. 17 Input Channels Specification

Each input day consists of 17 co-registered raster channels ($128 \times 128$ pixels):

| Channel Index | Band Name | Category | Description | Data Type / Normalization |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `pr` | Weather | Daily precipitation | Z-score normalized |
| **1** | `tmmx` | Weather | Maximum 2-meter air temperature | Z-score normalized |
| **2** | `tmmn` | Weather | Minimum 2-meter air temperature | Z-score normalized |
| **3** | `vs` | Weather | 10-meter wind speed | Z-score normalized |
| **4** | `th_sin` | Weather | Wind direction (sine component) | Z-score normalized |
| **5** | `th_cos` | Weather | Wind direction (cosine component) | Z-score normalized |
| **6** | `rmin` | Weather | Minimum relative humidity | Z-score normalized |
| **7** | `rmax` | Weather | Maximum relative humidity | Z-score normalized |
| **8** | `vpd` | Weather | Vapor Pressure Deficit (atmospheric drought) | Z-score normalized |
| **9** | `fm100` | Fuel | 100-hour dead fuel moisture | Z-score normalized |
| **10** | `erc` | Fire Danger | Energy Release Component | Z-score normalized |
| **11** | `NDVI` | Vegetation | Normalized Difference Vegetation Index | Z-score normalized |
| **12** | `EVI` | Vegetation | Enhanced Vegetation Index | Z-score normalized |
| **13** | `fire` | Target State | Binary active fire mask on previous days | Binary `[0.0, 1.0]` |
| **14** | `log1p_frp`| Fire Intensity| Fire Radiative Power ($\log(1 + \text{FRP})$) | Z-score normalized |
| **15** | `weather_valid`| Quality | Valid observation mask for weather channels | Binary `[0.0, 1.0]` |
| **16** | `vegetation_valid`| Quality | Valid observation mask for vegetation channels | Binary `[0.0, 1.0]` |

---

## 4. Dataset Splits

The dataset uses a strict temporal split by year to evaluate real-world generalization:

| Split | Years | Sequences | Percentage | Active Fire Sequences |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 2017 – 2021 | 42,622 | 70.2% | 27,370 (64.2%) |
| **Validation** | 2022 | 8,333 | 13.7% | 4,737 (56.9%) |
| **Test** | 2023 | 9,757 | 16.1% | 5,412 (55.5%) |
| **Total** | | **60,712** | **100%** | |

---

## 5. Loss Functions & Handling Extreme Imbalance

Active fire pixels typically constitute less than $1\%$ of the total pixels in a tile. To overcome severe class imbalance, the model is trained with **CombinedLoss**:

$$\mathcal{L}_{\text{total}} = w_{\text{focal}} \mathcal{L}_{\text{focal}} + w_{\text{dice}} \mathcal{L}_{\text{dice}}$$

1. **Binary Focal Loss:** Down-weights easy negative background pixels to focus gradient updates on sparse fire pixels:
   $$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t) \quad (\alpha=0.75, \, \gamma=2.0)$$
2. **Soft Dice Loss:** Directly maximizes spatial overlap and F1-score:
   $$\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum p y + \epsilon}{\sum p^2 + \sum y^2 + \epsilon} \quad (\epsilon=1.0)$$

---

## 6. Persistence Baseline Benchmark

In wildfire forecasting, the **Persistence Baseline** is the standard benchmark. It predicts that tomorrow's fire perimeter will be identical to today's fire perimeter:

$$\hat{Y}_{t+1} = Y_t$$

A valid machine learning model must outperform this baseline in **IoU (Jaccard Index)** and **F1-Score (Dice Coefficient)**.

---

## 7. Project Structure

```
wildfire-prediction-utae/
├── data/
│   └── processed_v1/
│       ├── train/              # 2017-2021 sequences (.npz + index.csv)
│       ├── val/                # 2022 validation sequences
│       ├── test/               # 2023 evaluation test set
│       ├── stats.json          # Z-score normalization statistics
│       ├── config.json         # Dataset processing configuration
│       └── quality_audit.csv   # Data audit log
│
├── src/
│   ├── utae_backbone/          # U-TAE reference modules (L-TAE, PosEnc, Convolutions)
│   ├── wildfire_dataset.py     # WildfireDataset and DataLoader with LRU caching
│   ├── model.py                # WildfireUTAE wrapper (17 channels -> 1 channel mask)
│   ├── losses.py               # FocalLoss, DiceLoss, CombinedLoss, MetricTracker
│   ├── test_batch.py           # Sanity check verifying shapes, NaN/Inf, forward & backward
│   ├── train.py                # Training script with checkpoints and metric curves
│   ├── evaluate.py             # Evaluation on 2023 test set vs. Persistence baseline
│   └── visualize.py            # Multi-panel prediction maps and attention plots
│
├── checkpoints/                # Best model checkpoint (best_model.pth)
├── outputs/                    # Visualizations, training curves, and evaluation reports
├── app.py                      # Interactive Gradio web demo (HF ZeroGPU compatible)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 8. Installation & Quickstart

### 8.1 Setup Environment

```bash
# Clone and enter directory
cd wildfire-prediction-utae

# Install dependencies
pip install -r requirements.txt
```

### 8.2 Run Batch Sanity Check
Verifies tensor dimensions, dtypes, NaN/Inf checks, forward pass, CombinedLoss, and backpropagation on real training data:

```bash
python src/test_batch.py
```

### 8.3 Training the Model
Trains WildfireUTAE on the training set (2017–2021) and monitors validation metrics (2022). Saves the best model to `checkpoints/best_model.pth` and curves to `outputs/training_curves.png`:

```bash
# Full training on GPU/CPU
python src/train.py --epochs 10 --batch_size 16 --lr 1e-3

# Or quick trial with subsampled data
python src/train.py --epochs 3 --batch_size 8 --max_train_samples 1000 --max_val_samples 200
```

### 8.4 Evaluation on Held-Out 2023 Test Set
Evaluates the trained model against the Persistence Baseline on the 2023 test set, performs a decision threshold sweep, and saves `outputs/baseline_comparison.csv` and `outputs/model_vs_baseline.png`:

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```

### 8.5 Visualizing Prediction Maps
Generates comprehensive multi-panel plots showing historical fire evolution, weather rasters, probability maps, error maps (TP, FP, FN), and temporal attention weights:

```bash
python src/visualize.py
```

### 8.6 Launching the Web Demo
Starts the interactive Gradio demo locally or on Hugging Face Spaces:

```bash
python app.py
```

---

## 9. References & Acknowledgments

- **U-TAE (Repository & Paper):**
  - Repository: [https://github.com/VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps)
  - Citation: Vivien Sainte Fare Garnot, Loic Landrieu. *Panoptic Segmentation of Satellite Image Time Series with U-TAE.* Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.
- This project integrates the U-TAE spatio-temporal architecture and L-TAE temporal self-attention mechanism for next-day wildfire spread prediction. Please refer to the original repository and comply with its MIT license.

