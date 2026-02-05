# BES-ELMnet Zenodo Distribution Manifest

## Files to Include in Zenodo Upload

### Core Files (REQUIRED)
```
1. bes_elmnet_inference.py         [Python script - main executable]
2. bes_data_4x16_v2.h5            [HDF5 data file - ~23 MB]
3. pre_modelV2.pt                  [PyTorch model - ~1-2 MB]
```

### Documentation (REQUIRED)
```
4. README.md                       [Main documentation - Zenodo landing page]
5. QUICKSTART.md                   [Quick start guide for users]
6. requirements.txt                [Python dependencies]
```

### Optional (RECOMMENDED)
```
7. LICENSE.txt                     [License file - e.g., MIT, CC BY 4.0]
8. CITATION.cff                    [Citation metadata file]
9. example_output.png              [Example of bes_elmnet_predictions.png]
```

---

## Total Package Size

**Estimated:** ~25-30 MB

Breakdown:
- `bes_data_4x16_v2.h5`: ~23 MB (largest file)
- `pre_modelV2.pt`: ~1-2 MB
- All code + documentation: <1 MB

---

## File Descriptions

### 1. bes_elmnet_inference.py
**Purpose:** Main executable script
**Size:** ~30 KB
**Description:** 
- Loads pre-trained model and HDF5 data
- Performs channel selection based on boundary
- Interpolates 4×8 to 8×8 signals
- Runs CNN predictions
- Generates comprehensive visualization

**Key features:**
- No signal generation (uses existing data)
- No training code (uses pre-trained model)
- Clean, production-ready inference pipeline
- Well-documented with inline comments

---

### 2. bes_data_4x16_v2.h5
**Purpose:** Plasma geometry and BES signal data
**Size:** ~23 MB
**Format:** HDF5

**Contents:**
```
geometry/
  ├── boundary (200, 2)           # D-shaped plasma LCFS in meters
  ├── rpos (64, 1)                # BES R positions in meters
  └── zpos (64, 1)                # BES Z positions in meters

channel_selection_validation/
  ├── selected_indices (32,)      # Validation: 4×8 selection
  └── radial_indices (8,)         # Validation: radial columns

signals/
  ├── time (45000,)               # Time array
  ├── all_signals (64, 45000)     # BES data from all channels
  └── attributes
      ├── sampling_rate: 1e6      # 1 MHz
      ├── duration: 0.045         # 45 ms
      └── n_channels: 64

elm_info/
  └── elm_positions (3,)          # ELM occurrence times in seconds
```

**Data characteristics:**
- Sampling rate: 1 MHz
- Duration: 45 ms (45,000 samples)
- Channels: 64 (4 poloidal × 16 radial)
- ELMs: 3 events
- Band-pass filtered: 15-150 kHz

---

### 3. pre_modelV2.pt
**Purpose:** Pre-trained BES-ELMnet CNN model
**Size:** ~1-2 MB
**Format:** PyTorch checkpoint

**Contents:**
```python
{
    'model_state_dict': OrderedDict(...),  # Model weights
    'optimizer_state_dict': {...},         # Optimizer state
    'epoch': 100,                          # Training epochs
    'train_losses': [...],                 # Training history
    'val_losses': [...],                   # Validation history
    'config': {...}                        # Configuration parameters
}
```

**Model architecture:**
- Input: 8×8×128 (spatial × temporal)
- 5 convolutional layers
- Total parameters: ~141,000
- Output: ELM onset probability [0, 1]

**Training:**
- Epochs: 100
- Optimizer: Adam (lr=0.001)
- Loss: MSE (Mean Squared Error)
- Dataset: DIII-D tokamak data

---

### 4. README.md
**Purpose:** Main documentation for Zenodo landing page
**Size:** ~15-20 KB
**Format:** Markdown

**Sections:**
- Overview and citation
- Quick start instructions
- Detailed workflow explanation
- Data specifications
- CNN architecture details
- Customization guide
- Scientific background
- Troubleshooting
- Acknowledgments

---

### 5. QUICKSTART.md
**Purpose:** Rapid onboarding guide
**Size:** ~8-10 KB
**Format:** Markdown

**Sections:**
- 5-minute installation
- 1-minute execution
- Understanding results
- Common troubleshooting
- Performance metrics
- Customization tips

---

### 6. requirements.txt
**Purpose:** Python package dependencies
**Size:** <1 KB
**Format:** Plain text

**Dependencies:**
```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
h5py>=3.8.0
```

---

## Zenodo Metadata

### Required Fields

**Title:**
```
BES-ELMnet: Pre-trained Model and Dataset for ELM Onset Prediction in Tokamaks
```

**Authors:**
```
[Your names and affiliations]
```

**Description:**
```
Convolutional neural network (CNN) for predicting Edge Localized Mode (ELM) 
onsets in tokamak fusion plasmas using Beam Emission Spectroscopy (BES) data.

Includes:
- Pre-trained PyTorch model (BES-ELMnet)
- HDF5 dataset with plasma geometry and BES signals (4×16 configuration)
- Python inference script for running predictions
- Complete documentation and quick start guide

Based on the methodology published in:
Joung et al. (2024), Nuclear Fusion 64:066038
https://doi.org/10.1088/1741-4326/ad43fb
```

**Keywords:**
```
tokamak, fusion plasma, ELM prediction, neural network, machine learning, 
edge localized mode, beam emission spectroscopy, plasma physics, DIII-D
```

**License:**
```
[Your choice - recommended: MIT or CC BY 4.0]
```

**Communities:**
```
- Fusion Energy
- Machine Learning
- Plasma Physics
```

**Grants/Funding:**
```
U.S. Department of Energy, Office of Science
- DE-FC02-04ER54698
- DE-SC0021157
- DE-SC0001288
- DE-FG02-08ER54999
```

**Related Identifiers:**
```
- isPreviousVersionOf: [future versions if applicable]
- isSupplementTo: https://doi.org/10.1088/1741-4326/ad43fb (paper)
- isDocumentedBy: [GitHub repo if applicable]
```

---

## Upload Checklist

Before uploading to Zenodo:

- [ ] Verify all core files present
- [ ] Test `bes_elmnet_inference.py` executes correctly
- [ ] Confirm HDF5 file loads without errors
- [ ] Verify model file loads and produces predictions
- [ ] Check README.md renders correctly on Zenodo
- [ ] Update DOI placeholders in documentation
- [ ] Add LICENSE.txt file
- [ ] Include example output image
- [ ] Fill in all Zenodo metadata fields
- [ ] Preview the Zenodo landing page
- [ ] Test download and execution on clean environment

---

## Post-Upload Tasks

After Zenodo publication:

1. **Update DOI badges** in README.md with actual DOI
2. **Update citations** in documentation
3. **Announce** on relevant channels (if applicable)
4. **Link** from paper supplementary materials
5. **Create GitHub release** (if using GitHub)

---

## Version Control

For future updates:

- Use Zenodo versioning system
- Document changes in a CHANGELOG.md
- Increment version numbers consistently
- Maintain backward compatibility when possible

**Version numbering:**
- v1.0: Initial release
- v1.1: Minor updates (bug fixes, documentation)
- v2.0: Major updates (new model, different architecture)

---

## Quality Assurance

### Pre-upload Testing

Test on clean environment:
```bash
# Create fresh conda environment
conda create -n test_bes python=3.10
conda activate test_bes

# Install requirements
pip install -r requirements.txt

# Run inference
python bes_elmnet_inference.py

# Verify output
ls -lh bes_elmnet_predictions.png
```

### Expected Results
- Execution completes without errors
- Output PNG file generated (~5-10 MB)
- Console output shows proper validation
- Predictions in expected range [0, 1]

---

## Distribution Summary

**What users get:**
1. Ready-to-run inference code
2. Pre-trained model (no training needed)
3. Example dataset (real DIII-D geometry and signals)
4. Comprehensive documentation
5. Quick start guide

**What users can do:**
- Run predictions on provided data
- Understand the methodology
- Adapt code for their own data
- Use pre-trained model as starting point
- Reproduce paper results

**What's NOT included:**
- Training code (focuses on inference)
- Signal generation code (uses real data)
- Raw DIII-D data (processed and ready to use)
- Multiple shots (one representative example)

---

## File Integrity

### Checksums (Generate before upload)

```bash
# Generate MD5 checksums
md5sum bes_elmnet_inference.py >> checksums.txt
md5sum bes_data_4x16_v2.h5 >> checksums.txt
md5sum pre_modelV2.pt >> checksums.txt

# Generate SHA256 checksums
sha256sum bes_elmnet_inference.py >> checksums_sha256.txt
sha256sum bes_data_4x16_v2.h5 >> checksums_sha256.txt
sha256sum pre_modelV2.pt >> checksums_sha256.txt
```

Include checksums.txt in the distribution for verification.

---

**Distribution ready for Zenodo upload!** 🎉
