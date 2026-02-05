# Prediction

**Physics-consistent input adaptation for Convolutional neural network**

---

## 📦 Contents

This distribution contains:

1. **`bes_nn_inference.py`** - Python script for running predictions
2. **`bes_data_4x16_v2.h5`** - HDF5 file with plasma geometry and signals
3. **`pre_modelV2.pt`** - Pre-trained PyTorch CNN model
4. **`README.md`** - This file

---

## 🚀 Quick Start

### Requirements

```bash
pip install torch numpy scipy matplotlib h5py
```

**Tested with:**
- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- SciPy 1.10+
- matplotlib 3.7+
- h5py 3.8+

### Running the Code

```bash
python bes_nn_inference.py
```

**Output:** `nn_predictions.png` - Comprehensive visualization with 9 analysis panels

---

## 📊 What the Code Does

### Workflow

```
1. Load HDF5 Data
   └─ Plasma boundary (LCFS)
   └─ 4×16 channel positions (64 total)
   └─ Signal data

2. Select 4×8 Channels
   └─ Calculate from boundary location
   └─ Always recalculated (not loaded from file)
   └─ Validate against saved selection

3. Interpolate 4×8 → 8×8
   └─ Linear interpolation along poloidal direction
   └─ Required for CNN input

4. Normalize & Window
   └─ Per-microsecond normalization
   └─ 128 µs sliding windows
   └─ 32 µs stride

5. Load Pre-trained Model
   └─ CNN architecture

6. Run Predictions
   └─ Onset probabilities [0, 1]
   └─ Threshold: 0.6

7. Visualize Results
   └─ Comprehensive 9-panel figure
```

---

## 🔧 Data Specifications

### HDF5 File Structure

```
bes_data_4x16_v2.h5
├── geometry/
│   ├── boundary (200, 2)          # Single LCFS in meters
│   ├── rpos (64, 1)               # R positions in meters
│   └── zpos (64, 1)               # Z positions in meters
├── channel_selection_validation/
│   ├── selected_indices (32,)     # For validation only
│   └── radial_indices (8,)        # For validation only
├── signals/
│   ├── time (45000,)              # Time array
│   └── all_signals (64, 45000)    # BES signals from all channels
└── elm_info/
    └── elm_positions (3,)          # ELM occurrence times
```

### Channel Configuration

- **Total channels:** 64 (4 poloidal × 16 radial)
- **Selected channels:** 32 (4 poloidal × 8 radial)
- **Sampling rate:** 1 MHz
- **Duration:** 45 ms
- **Band-pass filter:** 15-150 kHz

---

## 🧠 CNN Architecture

**BESNN** follows the paper architecture:

```
Input: 8 × 8 × 128 (spatial × temporal)
├─ Conv2D: 8 → 64 channels (3×3 kernel, stride=1)
├─ ReLU + MaxPool2D (÷2)
├─ Conv2D: 64 → 32 channels (3×3 kernel, stride=1)
├─ ReLU + MaxPool2D (÷2)
├─ Conv2D: 32 → 16 channels (3×3 kernel, stride=1)
├─ ReLU + MaxPool2D (÷2)
├─ Conv2D: 16 → 128 channels (1×16 kernel, stride=1)
├─ ReLU
├─ Conv2D: 128 → 1 channel (1×1 kernel, stride=1)
└─ Sigmoid → Output: ELM onset probability [0, 1]
```

**Total parameters:** ~141,000

---

## 📈 Output Visualization

The script generates `bes_nn_predictions.png` with 9 panels:

1. **Channel Selection** - All 4×16 channels with selected 4×8 highlighted
2. **Zoomed View** - Detailed view of selected channels vs plasma boundary
3. **4×8 Signal Heatmap** - Selected signals over time
4. **Interpolation Demo** - Shows 4→8 interpolation for one radial column
5. **8×8 Signal Heatmap** - Interpolated signals ready for CNN
6. **Signal Comparison** - Original vs interpolated time series
7. **CNN Predictions** - Onset probabilities with threshold
8. **Statistics** - Execution summary and detection metrics
9. **Workflow** - Step-by-step processing diagram

---

## 🔑 Key Features

### 1. Boundary-Based Channel Selection

Channels are selected based on plasma boundary crossing:

```python
# Find where boundary intersects midplane region
bdryindex = np.where((bdry_r > 2.0) & 
                     (bdry_z > -0.1) & 
                     (bdry_z < 0.1))[0]

# Select 8 radial channels spanning boundary
# (some inside, some outside)
```

This ensures the selected channels capture edge physics relevant to ELM onset.

### 2. Poloidal Interpolation

Interpolates from 4 poloidal channels to 8 using linear interpolation:

```
Original 4×8:          Interpolated 8×8:
Z₀ ○○○○○○○○          Z₀ ○○○○○○○○
                      Z₁ ●●●●●●●● (interpolated)
Z₁ ○○○○○○○○          Z₂ ○○○○○○○○
                      Z₃ ●●●●●●●● (interpolated)
Z₂ ○○○○○○○○          Z₄ ○○○○○○○○
                      Z₅ ●●●●●●●● (interpolated)
Z₃ ○○○○○○○○          Z₆ ○○○○○○○○
                      Z₇ ●●●●●●●● (interpolated)
```

This provides the 8×8 spatial grid required by the CNN.

---

## 📝 Customization

### Change Input Files

Edit lines 21-22 in `bes_nn_inference.py`:

```python
HDF5_FILE = 'your_data.h5'
MODEL_FILE = 'your_model.pt'
```

### Change Window Parameters

Edit lines 24-25:

```python
WINDOW_SIZE = 128  # microseconds
STRIDE = 32        # microseconds
```
---

## 🤝 Contributing

This is a reference implementation for the published methodology. For questions or issues:

- **Paper:** https://doi.org/10.1088/1741-4326/ad43fb

---

## 📄 License

[Specify your license - e.g., MIT, CC BY 4.0, etc.]

---
## 📋 Version History

- **v1.0** (2024) - Initial release
  - Pre-trained model
  - 4×16 BES configuration
  - Boundary-based channel selection
  - Linear poloidal interpolation

---
