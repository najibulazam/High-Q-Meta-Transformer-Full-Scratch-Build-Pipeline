# High-Q Meta-Transformer: Full Scratch-Build Pipeline

This repository contains an end-to-end automated framework for generating, simulating, and designing high-Q quasi-BIC (Bound States in the Continuum) metasurfaces from scratch.

Unlike conventional workflows that rely on a pre-configured Lumerical project, this framework builds the complete 3D FDTD simulation entirely through Python scripting. It also employs 1D Vision Transformers (ViT) for ultra-fast forward spectrum prediction and inverse geometry reconstruction.

---

# Architecture & Workflow

The project is organized into independent modules for automated FDTD scene generation, dataset creation, transformer training, and inverse metasurface design.

## Structural Flow

### 1. Scene Generation (`fdtd/`)

Automatically:

- Launches a fresh Lumerical FDTD session
- Creates a 3D simulation region
- Configures Periodic and PML boundary conditions
- Builds a **CaF₂ substrate**
- Places a rotated **Germanium elliptical resonator**
- Adds an oblique plane-wave source
- Deploys transmission/reflection monitors

---

### 2. Deep Learning Pipeline (`model/` & `train/`)

#### SpecViT (Forward Model)

Maps the five geometric parameters

- **H**
- **Px**
- **Py**
- **R1**
- **R2**

into a **300-point transmission/reflection spectrum**.

#### MetaViT (Inverse Model)

Predicts the geometric parameters from a desired spectrum using a **dual-loss optimization**:

- Geometry Mean Squared Error (MSE)
- Spectrum Consistency Loss (through a frozen, pre-trained SpecViT)

This ensures that the predicted geometry not only matches the target parameters but also reproduces the desired electromagnetic response.

---

### 3. One-Shot Meta-Design (`design/`)

Accepts arbitrary target spectra (such as high-Q Fano resonance profiles) and instantly predicts the corresponding nanoscale geometry.

---

# Directory Structure

```text
high_q_meta_transformer/
│
├── config.yaml                     # Global configuration and hyperparameters
├── requirements.txt                # Python dependencies
│
├── fdtd/
│   ├── setup_qBIC_cell.py          # Builds the complete 3D FDTD model from scratch
│   └── run_spectral_sweep.py       # Automated spectral data generation
│
├── model/
│   ├── spec_vit.py                 # SpecViT Forward Transformer
│   └── meta_vit.py                 # MetaViT Inverse Transformer
│
├── train/
│   ├── spectral_dataset.py         # Dataset loader & preprocessing
│   ├── train_spec_vit.py           # Forward model training
│   └── train_meta_vit.py           # Inverse model training
│
├── design/
│   ├── target_spectrum.py          # Generate synthetic target spectra
│   └── predict_geometry.py         # Predict geometry from spectrum
│
└── README.md
```

---

# Detailed Execution Guide

Execute the following modules in sequence.

---

## Step 1 — Automated Data Collection

Generate the simulation dataset from scratch.

```bash
python -m fdtd.run_spectral_sweep
```

### Execution Details

The script automatically:

- Launches Lumerical FDTD using `lumapi`
- Clears the workspace
- Builds a new unit cell using randomized geometric parameters
- Runs the electromagnetic simulation
- Extracts the transmission/reflection spectrum
- Saves checkpoints every **5 iterations**

Generated dataset:

```text
dataset/qBIC_spectral_data.csv
```

---

## Step 2 — Train SpecViT (Forward Model)

Train the forward Vision Transformer.

```bash
python -m train.train_spec_vit
```

### Execution Details

The training pipeline:

- Loads the generated dataset
- Fits a `StandardScaler` to the geometry parameters
- Saves the scaler as

```text
saved_models/scaler_geo.pkl
```

- Trains SpecViT using Mean Squared Error (MSE)
- Saves the best-performing model

```text
saved_models/best_spec_vit.pth
```

---

## Step 3 — Train MetaViT (Inverse Model)

Train the inverse transformer.

```bash
python -m train.train_meta_vit
```

### Execution Details

The training script:

- Loads the pre-trained SpecViT
- Freezes its weights
- Uses SpecViT as a spectral consistency evaluator
- Optimizes MetaViT using

  - Geometry Loss (MSE)
  - Spectrum Consistency Loss

Best model:

```text
saved_models/best_meta_vit.pth
```

---

## Step 4 — Generate a Target Spectrum

Create an artificial high-Q target spectrum.

```bash
python -m design.target_spectrum
```

### Execution Details

The script generates:

- Narrow linewidth
- High-Q Fano resonance
- Mid-infrared target spectrum

Output:

```text
dataset/target_spectrum.csv
```

---

## Step 5 — Instant Layout Synthesis

Predict the required geometry.

```bash
python -m design.predict_geometry
```

### Execution Details

The script:

- Loads `target_spectrum.csv`
- Passes it through MetaViT
- Applies the inverse scaling transformation
- Prints the predicted values

Output:

```text
H
Px
Py
R1
R2
```

These values can be directly used to rebuild the metasurface inside Lumerical.

---

# Command & Operational Reference

| Command | Action | Output |
|----------|--------|--------|
| `python -m fdtd.run_spectral_sweep` | Builds and simulates randomized FDTD structures | `dataset/qBIC_spectral_data.csv` |
| `python -m train.train_spec_vit` | Trains the forward Vision Transformer | `saved_models/best_spec_vit.pth` & `saved_models/scaler_geo.pkl` |
| `python -m train.train_meta_vit` | Trains the inverse Vision Transformer | `saved_models/best_meta_vit.pth` |
| `python -m design.target_spectrum` | Generates a synthetic high-Q target spectrum | `dataset/target_spectrum.csv` |
| `python -m design.predict_geometry` | Predicts the optimal geometry from the target spectrum | Prints predicted `H`, `Px`, `Py`, `R1`, and `R2` |

---

# Configuration (`config.yaml`)

Project parameters can be modified through `config.yaml`.

## Dataset

```yaml
dataset:
  num_samples: 5000
```

Adjust the total number of simulated structures.

---

## Model Hyperparameters

```yaml
model:
  learning_rate: 0.0001
  embed_dim: 128
  epochs: 200
  batch_size: 64
  num_heads: 8
  depth: 6
```

Common parameters include:

- `learning_rate`
- `batch_size`
- `embed_dim`
- `num_heads`
- `depth`
- `epochs`

---

# Complete Pipeline Overview

```text
           Random Parameters
                   │
                   ▼
      setup_qBIC_cell.py
                   │
                   ▼
        Lumerical FDTD Simulation
                   │
                   ▼
     run_spectral_sweep.py
                   │
                   ▼
     qBIC_spectral_data.csv
                   │
          ┌────────┴────────┐
          ▼                 ▼
     train_spec_vit    train_meta_vit
          │                 │
          ▼                 ▼
    best_spec_vit      best_meta_vit
          │                 │
          └────────┬────────┘
                   ▼
      target_spectrum.py
                   │
                   ▼
      predict_geometry.py
                   │
                   ▼
 Predicted Geometry (H, Px, Py, R1, R2)
```

---

# Generated Files

After running the complete pipeline, the following artifacts will be available:

```text
dataset/
├── qBIC_spectral_data.csv
└── target_spectrum.csv

saved_models/
├── best_spec_vit.pth
├── best_meta_vit.pth
└── scaler_geo.pkl
```

---

# Requirements

Recommended software stack:

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- PyYAML
- tqdm
- Lumerical FDTD
- Lumerical Python API (`lumapi`)

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

---

# Citation

If you use this project in academic research, please cite the associated publication describing the Transformer-based high-Q quasi-BIC metasurface inverse design methodology.

---

# License

This project is intended for research, educational, and academic purposes.