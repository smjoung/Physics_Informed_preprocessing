# BES-ELMnet Quick Start Guide

## 🚀 Installation (5 minutes)

### Step 1: Install Python Packages

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch numpy scipy matplotlib h5py
```

### Step 2: Verify Files

Ensure you have these files in your directory:

```
your_directory/
├── bes_elmnet_inference.py     ← Main script
├── bes_data_4x16_v2.h5         ← Data file (~23 MB)
├── pre_modelV2.pt              ← Model file (~1-2 MB)
├── requirements.txt
└── README.md
```

---

## ▶️ Run the Code (1 minute)

```bash
python bes_elmnet_inference.py
```

**What you'll see:**

```
======================================================================
BES-ELMnet: ELM Onset Prediction
======================================================================
HDF5 file: bes_data_4x16_v2.h5
Model file: pre_modelV2.pt
======================================================================

Device: cpu

Loading data from bes_data_4x16_v2.h5...
✓ Loaded geometry and signals
  - Boundary shape: (200, 2) (single LCFS in meters)
  - BES positions: (64, 1) in meters
  - Signal samples: 45000
  - Sampling rate: 1.0 MHz
  - Number of ELMs: 3

======================================================================
CALCULATING CHANNEL SELECTION FROM BOUNDARY
======================================================================
Boundary reference R: 2.2154 m
Found 48 boundary points in selection region
BES radial range: 2.1500 to 2.3000 m
Boundary crossing at radial index: 8
Selected radial indices: [2 3 4 5 6 7 8 9]
Selected radial R range: 2.1600 to 2.2400 m
Total selected channels: 32 (4×8)
======================================================================

======================================================================
VALIDATING CALCULATED SELECTION AGAINST SAVED
======================================================================
✓ VALIDATION PASSED: Calculated selection matches saved selection
======================================================================

Extracted 4×8 signals: (32, 45000)
Interpolating 4×8 → 8×8...
Interpolated to 8×8: (64, 45000)
Normalizing signals...
Preparing windows for model...
Created 1403 windows

Loading model from pre_modelV2.pt...
✓ Model loaded successfully
  Trained for 100 epochs
  Final train loss: 0.0234

Running predictions...
✓ Predictions complete
  Prediction range: [0.123, 0.876]

Generating visualizations...
✓ Figure saved: bes_elmnet_predictions.png

======================================================================
EXECUTION COMPLETE!
======================================================================
Output: bes_elmnet_predictions.png
======================================================================
```

**Runtime:** 5-15 seconds (CPU) or 2-5 seconds (GPU)

---

## 📊 Output

### Generated File

**`bes_elmnet_predictions.png`** - Comprehensive visualization with 9 panels:

1. Channel selection on plasma
2. Zoomed view of selected channels
3. 4×8 signal heatmap
4. Interpolation demonstration
5. 8×8 signal heatmap (for CNN)
6. Signal comparison plot
7. **CNN predictions vs time** ← Main result!
8. Statistics summary
9. Workflow diagram

---

## 🔍 Understanding the Results

### Panel 7: CNN Predictions

Look for:
- **Blue line:** ELM onset probability (0 to 1)
- **Green dashed line:** Threshold (0.601)
- **Red dashed lines:** Actual ELM times

**Interpretation:**
- Probability **above threshold** → ELM predicted
- Prediction **before red line** → Successful early warning
- Typical warning time: **2-5 milliseconds** before ELM

### Example

```
Time: 9.5 ms  → Probability: 0.45 (below threshold, safe)
Time: 9.8 ms  → Probability: 0.72 (ABOVE threshold, ELM predicted!)
Time: 10.0 ms → Actual ELM occurs ✓
```

**Success!** 0.2 ms early warning

---

## 🛠️ Troubleshooting

### "ERROR: HDF5 file not found"

**Solution:** Make sure `bes_data_4x16_v2.h5` is in the same directory as the script.

```bash
ls -lh bes_data_4x16_v2.h5
# Should show: -rw-r--r--  1 user  staff   23M ...
```

### "ERROR: Model file not found"

**Solution:** Make sure `pre_modelV2.pt` is in the same directory.

```bash
ls -lh pre_modelV2.pt
# Should show: -rw-r--r--  1 user  staff   1.8M ...
```

### "WARNING: PyTorch not available"

**Solution:** Install PyTorch:

```bash
pip install torch
```

For GPU support (optional, faster):

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Slow execution?

**On CPU:** 10-20 seconds is normal

**Speed up with GPU:**
1. Install CUDA-enabled PyTorch (see above)
2. Verify GPU detected: Script will print `Device: cuda`
3. Expected speedup: 5-10×

---

## 🔧 Customization

### Use Your Own Data

1. **Prepare HDF5 file** with structure:
   ```
   your_data.h5
   ├── geometry/boundary (n, 2) in meters
   ├── geometry/rpos (64, 1) in meters
   ├── geometry/zpos (64, 1) in meters
   ├── signals/all_signals (64, n_samples)
   └── elm_info/elm_positions (n_elms,)
   ```

2. **Edit script** (line 21):
   ```python
   HDF5_FILE = 'your_data.h5'
   ```

3. **Run:**
   ```bash
   python bes_elmnet_inference.py
   ```

### Use Your Own Model

1. **Train model** (see paper for methodology)

2. **Save as .pt file:**
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'epoch': n_epochs,
       'train_losses': losses
   }, 'your_model.pt')
   ```

3. **Edit script** (line 22):
   ```python
   MODEL_FILE = 'your_model.pt'
   ```

### Adjust Threshold

**Higher threshold (0.7)** → Fewer false alarms, might miss some ELMs
**Lower threshold (0.5)** → Catch more ELMs, more false alarms

Edit line 26:
```python
PREDICTION_THRESHOLD = 0.601  # Try 0.5 to 0.7
```

---

## 📈 Performance Metrics

**Typical results with DIII-D data:**
- **Precision:** 70-90%
- **Recall:** 85-95%
- **Warning time:** 2-5 ms before ELM
- **False alarm rate:** 5-15%

---

## 💡 Tips

1. **First run:** Will take longer (~15 sec) as Python loads libraries
2. **Subsequent runs:** Faster (~5-10 sec) with cached imports
3. **Batch processing:** Process multiple shots by looping over HDF5 files
4. **Real-time:** For tokamak operations, implement on FPGA/GPU

---

## 📚 Next Steps

1. **Read the paper:** Joung et al. 2024, Nuclear Fusion
2. **Explore the code:** Well-commented, modify as needed
3. **Try your data:** Follow HDF5 structure guidelines
4. **Cite the work:** See README.md for BibTeX

---

## 🆘 Need Help?

1. **Check README.md** for detailed documentation
2. **Review code comments** in `bes_elmnet_inference.py`
3. **Read the paper** for methodology details
4. **Contact authors** (see README.md)

---

**Ready to go! Run the code and explore ELM prediction.** 🚀
