# BES-ELMnet: Edge Localized Mode Onset Prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**Convolutional neural network for predicting Edge Localized Mode (ELM) onsets in tokamak plasmas using Beam Emission Spectroscopy (BES) data.**

Based on the methodology from:
> Joung, S., Smith, D.R., McKee, G., et al. (2024). Tokamak edge localized mode onset prediction with deep neural network and pedestal turbulence. *Nuclear Fusion*, 64(6), 066038. https://doi.org/10.1088/1741-4326/ad43fb

---

## 📦 Contents

This distribution contains:

1. **`bes_elmnet_inference.py`** - Python script for running ELM predictions
2. **`bes_data_4x16_v2.h5`** - HDF5 file with plasma geometry and BES signals
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
python bes_elmnet_inference.py
```

**Output:** `bes_elmnet_predictions.png` - Comprehensive visualization with 9 analysis panels

---

## 📊 What the Code Does

### Workflow

```
1. Load HDF5 Data
   └─ Plasma boundary (LCFS)
   └─ 4×16 BES channel positions (64 total)
   └─ BES signal data

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
   └─ BES-ELMnet CNN architecture
   └─ Trained on DIII-D data

6. Run Predictions
   └─ ELM onset probabilities [0, 1]
   └─ Threshold: 0.601

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

### BES Configuration

- **Total channels:** 64 (4 poloidal × 16 radial)
- **Selected channels:** 32 (4 poloidal × 8 radial)
- **Sampling rate:** 1 MHz
- **Duration:** 45 ms
- **Band-pass filter:** 15-150 kHz

---

## 🧠 CNN Architecture

**BES-ELMnet** follows the paper architecture:

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

The script generates `bes_elmnet_predictions.png` with 9 panels:

1. **BES Channel Selection** - All 4×16 channels with selected 4×8 highlighted
2. **Zoomed View** - Detailed view of selected channels vs plasma boundary
3. **4×8 Signal Heatmap** - Selected BES signals over time
4. **Interpolation Demo** - Shows 4→8 interpolation for one radial column
5. **8×8 Signal Heatmap** - Interpolated signals ready for CNN
6. **Signal Comparison** - Original vs interpolated time series
7. **CNN Predictions** - ELM onset probabilities with threshold
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

### 3. Real-time Capable

- **Window size:** 128 µs (very short)
- **Stride:** 32 µs (high temporal resolution)
- **Prediction:** Can forecast ELMs 2-5 ms in advance
- **FPGA-ready:** Architecture designed for hardware acceleration

---

## 📝 Customization

### Change Input Files

Edit lines 21-22 in `bes_elmnet_inference.py`:

```python
HDF5_FILE = 'your_data.h5'
MODEL_FILE = 'your_model.pt'
```

### Adjust Prediction Threshold

Edit line 26:

```python
PREDICTION_THRESHOLD = 0.601  # Adjust between 0.5 and 0.7
```

### Change Window Parameters

Edit lines 24-25:

```python
WINDOW_SIZE = 128  # microseconds
STRIDE = 32        # microseconds
```

---

## 🔬 Scientific Background

### Edge Localized Modes (ELMs)

ELMs are quasi-periodic instabilities in high-confinement mode (H-mode) tokamak plasmas that:
- Eject significant plasma energy (~20%) in ~100 µs
- Can damage plasma-facing components
- Are triggered by pressure gradient exceeding stability threshold
- Show precursor signatures in pedestal turbulence

### Why This Matters

Predicting ELMs enables:
- **Proactive control:** Apply mitigation (RMP coils) before onset
- **Reactor protection:** Prevent damage to first wall
- **Operational efficiency:** Maintain H-mode without disruptions

### BES System

Beam Emission Spectroscopy measures density fluctuations by observing:
- Neutral beam injection + background plasma → excited atoms
- Doppler-shifted Hα emission (656.1 nm)
- 2D spatial array (4×16 channels)
- High temporal resolution (1 MHz)
- Localized measurement in pedestal region

---

## 📚 Citation

If you use this code or data, please cite:

```bibtex
@article{joung2024tokamak,
  title={Tokamak edge localized mode onset prediction with deep neural network and pedestal turbulence},
  author={Joung, Semin and Smith, David R and McKee, G and Yan, Z and Gill, K and Zimmerman, J and Geiger, B and Coffee, R and O'Shea, FH and Jalalvand, A and Kolemen, E},
  journal={Nuclear Fusion},
  volume={64},
  number={6},
  pages={066038},
  year={2024},
  publisher={IOP Publishing}
}
```

**And this dataset:**

```bibtex
@dataset{[YOUR_DATASET_INFO],
  author       = {[Your Name]},
  title        = {BES-ELMnet: Pre-trained Model and Dataset},
  month        = [Month],
  year         = [Year],
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

---

## 🤝 Contributing

This is a reference implementation for the published methodology. For questions or issues:

- **Paper:** https://doi.org/10.1088/1741-4326/ad43fb
- **Contact:** [Your contact information]

---

## 📄 License

[Specify your license - e.g., MIT, CC BY 4.0, etc.]

**Data source:** DIII-D tokamak operated by General Atomics

---

## 🙏 Acknowledgments

- DIII-D Team at General Atomics
- University of Wisconsin-Madison
- SLAC National Accelerator Laboratory
- Princeton University
- U.S. Department of Energy, Office of Science

This work was supported by the U.S. Department of Energy under Awards DE-FC02-04ER54698, DE-SC0021157, DE-SC0001288, and DE-FG02-08ER54999.

---

## 📋 Version History

- **v1.0** (2024) - Initial release
  - Pre-trained model on DIII-D data
  - 4×16 BES configuration
  - Boundary-based channel selection
  - Linear poloidal interpolation

---

## ⚠️ Disclaimer

This is research code and data distributed for scientific reproducibility. For production use in tokamak operations:
- Validate thoroughly on your specific tokamak
- Implement real-time processing (FPGA/GPU)
- Integrate with control systems appropriately
- Test extensively in simulation before deployment

---

## 🔗 Related Resources

- **DIII-D:** https://www.ga.com/magnetic-fusion/diii-d
- **Nuclear Fusion Journal:** https://iopscience.iop.org/journal/0029-5515
- **PyTorch:** https://pytorch.org/
- **HDF5:** https://www.hdfgroup.org/

---

**For support or questions, please open an issue or contact the authors.**
