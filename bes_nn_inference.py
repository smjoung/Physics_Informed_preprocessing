#!/usr/bin/env python3
"""
BESNN: ELM Onset Prediction Using 4×16 BES Configuration
Loads pre-trained model and HDF5 data, runs predictions

For Zenodo distribution - requires:
  - bes_data_4x16_v2.h5 (HDF5 data file)
  - pre_modelV2.pt (trained PyTorch model)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
import h5py
import os
import sys

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("ERROR: PyTorch not available.")
    print("Install with: pip install torch")
    sys.exit(1)

np.random.seed(42)
torch.manual_seed(42)

# ==================== PARAMETERS ====================
# Files (user can modify these)
HDF5_FILE = 'bes_data_4x16_v2.h5'
MODEL_FILE = 'pre_modelV2.pt'

# CNN parameters (from paper)
WINDOW_SIZE = 128  # microseconds
STRIDE = 32  # microseconds
PREDICTION_THRESHOLD = 0.601  # From paper

print("="*70)
print("BESNN: ELM Onset Prediction")
print("="*70)
print(f"HDF5 file: {HDF5_FILE}")
print(f"Model file: {MODEL_FILE}")
print("="*70 + "\n")


# ==================== CHANNEL SELECTION ====================
def select_4x8_channels(boundary, rpos, zpos, n_radial=16):
    """
    Select 4×8 channels based on boundary location
    ALWAYS calculates from boundary - never uses saved selection
    
    Args:
        boundary: (n_points, 2) array - single LCFS in meters
        rpos: (64, 1) array of R positions in meters
        zpos: (64, 1) array of Z positions in meters
        n_radial: number of radial channels (default 16)
    
    Returns:
        selected_indices: indices of selected 4×8=32 channels
        radial_indices: the 8 radial column indices selected
    """
    print("\n" + "="*70)
    print("CALCULATING CHANNEL SELECTION FROM BOUNDARY")
    print("="*70)
    
    # boundary is (n_points, 2) - single surface in meters
    bdry_r = boundary[:, 0]  # R coordinates
    bdry_z = boundary[:, 1]  # Z coordinates
    
    # Find boundary index where R > 2.0 and -0.1 < Z < 0.1
    bdryindex = np.where((bdry_r > 2.0) & 
                         (bdry_z > -0.1) & 
                         (bdry_z < 0.1))[0]
    
    if len(bdryindex) == 0:
        print("Warning: No boundary points found in specified region. Using default.")
        bdry_r_ref = 2.2
    else:
        bdry_r_ref = np.mean(bdry_r[bdryindex])
    
    print(f"Boundary reference R: {bdry_r_ref:.4f} m")
    print(f"Found {len(bdryindex)} boundary points in selection region")
    
    # Find radial index where BES crosses boundary
    rpos_first_row = rpos[:n_radial, 0]  # First poloidal row (16 radial positions)
    
    print(f"BES radial range: {rpos_first_row.min():.4f} to {rpos_first_row.max():.4f} m")
    
    # Find first index where R > boundary reference
    rbes_index_array = np.where(rpos_first_row > bdry_r_ref)[0]
    
    if len(rbes_index_array) == 0:
        print("Warning: No BES channels outside boundary. Using default index.")
        rbes_index = 8  # Default to middle
    else:
        rbes_index = rbes_index_array[0]
    
    print(f"Boundary crossing at radial index: {rbes_index}")
    
    # Select 8 radial columns centered on boundary crossing
    rbes_chosen = np.arange(rbes_index - 6, rbes_index + 2, 1)  # 8 channels
    
    # Clip to valid range [0, n_radial-1]
    rbes_chosen = np.clip(rbes_chosen, 0, n_radial - 1)
    
    # Ensure exactly 8 unique channels
    rbes_chosen = np.unique(rbes_chosen)
    if len(rbes_chosen) < 8:
        # Extend if needed
        while len(rbes_chosen) < 8:
            if rbes_chosen[-1] < n_radial - 1:
                rbes_chosen = np.append(rbes_chosen, rbes_chosen[-1] + 1)
            else:
                rbes_chosen = np.insert(rbes_chosen, 0, rbes_chosen[0] - 1)
        rbes_chosen = np.unique(rbes_chosen)
    
    # Take first 8
    rbes_chosen = rbes_chosen[:8]
    
    print(f"Selected radial indices: {rbes_chosen}")
    print(f"Selected radial R range: {rpos_first_row[rbes_chosen[0]]:.4f} to {rpos_first_row[rbes_chosen[-1]]:.4f} m")
    
    # Determine number of poloidal channels
    n_poloidal = len(rpos) // n_radial
    
    # Select all poloidal positions for these 8 radial positions
    selected_indices = []
    for poloidal_idx in range(n_poloidal):
        for rad_idx in rbes_chosen:
            channel_idx = poloidal_idx * n_radial + rad_idx
            selected_indices.append(channel_idx)
    
    selected_indices = np.array(selected_indices)
    
    print(f"Total selected channels: {len(selected_indices)} ({n_poloidal}×8)")
    print("="*70 + "\n")
    
    return selected_indices, rbes_chosen


# ==================== HDF5 LOADING ====================
def load_from_hdf5(filename):
    """
    Load data from HDF5 file
    Note: Does NOT load channel selection - must be calculated fresh
    """
    if not os.path.exists(filename):
        print(f"ERROR: HDF5 file not found: {filename}")
        print("Please ensure the HDF5 file is in the current directory.")
        sys.exit(1)
    
    print(f"Loading data from {filename}...")
    
    with h5py.File(filename, 'r') as f:
        # Geometry
        boundary = f['geometry/boundary'][:]  # Single surface (n_points, 2) in meters
        rpos = f['geometry/rpos'][:]  # (64, 1) in meters
        zpos = f['geometry/zpos'][:]  # (64, 1) in meters
        
        # Load validation data for comparison (optional)
        if 'channel_selection_validation' in f:
            selected_indices_saved = f['channel_selection_validation/selected_indices'][:]
            radial_indices_saved = f['channel_selection_validation/radial_indices'][:]
        else:
            selected_indices_saved = None
            radial_indices_saved = None
        
        # Signals
        time_array = f['signals/time'][:]
        all_signals = f['signals/all_signals'][:]
        
        # ELM info
        elm_positions = f['elm_info/elm_positions'][:]
        
        # Get sampling rate and other metadata
        sampling_rate = f['signals'].attrs.get('sampling_rate', 1e6)
    
    print(f"✓ Loaded geometry and signals")
    print(f"  - Boundary shape: {boundary.shape} (single LCFS in meters)")
    print(f"  - BES positions: {rpos.shape} in meters")
    print(f"  - Signal samples: {all_signals.shape[1]}")
    print(f"  - Sampling rate: {sampling_rate/1e6:.1f} MHz")
    print(f"  - Number of ELMs: {len(elm_positions)}")
    print(f"  - NOTE: Channel selection will be calculated fresh from boundary\n")
    
    return (boundary, rpos, zpos, time_array, all_signals, elm_positions,
            selected_indices_saved, radial_indices_saved, sampling_rate)


# ==================== CNN MODEL ====================
class ConvNet(nn.Module):
    """BESNN architecture (corrected for dimension flow)"""
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 128, kernel_size=(1, 16), stride=1, padding=0)
        self.conv5 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        
        self.relu = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool2(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = self.relu(self.conv4(x))
        x = self.sigmoid(self.conv5(x))
        return x


def load_model(model_path, device):
    """Load trained model from .pt file"""
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please ensure the model file is in the current directory.")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    
    model = ConvNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs")
    if 'train_losses' in checkpoint and len(checkpoint['train_losses']) > 0:
        print(f"  Final train loss: {checkpoint['train_losses'][-1]:.4f}")
    print()
    
    return model, checkpoint


# ==================== INTERPOLATION ====================
def interpolate_4x8_to_8x8(signals_4x8):
    """
    Interpolate 4×8 signals to 8×8 using linear interpolation along poloidal (Z) direction
    
    Args:
        signals_4x8: (32, n_time) array - 4 poloidal × 8 radial × n_time
    
    Returns:
        signals_8x8: (64, n_time) array - 8 poloidal × 8 radial × n_time
    """
    n_time = signals_4x8.shape[1]
    n_radial = 8
    n_poloidal_in = 4
    n_poloidal_out = 8
    
    # Reshape to (4, 8, n_time)
    signals_reshaped = signals_4x8.reshape(n_poloidal_in, n_radial, n_time)
    
    # Create output array (8, 8, n_time)
    signals_interpolated = np.zeros((n_poloidal_out, n_radial, n_time))
    
    # Original poloidal indices (4 points)
    z_original = np.arange(n_poloidal_in)
    
    # Target poloidal indices (8 points)
    z_target = np.linspace(0, n_poloidal_in - 1, n_poloidal_out)
    
    # Interpolate for each radial position and each time point
    for r_idx in range(n_radial):
        for t_idx in range(n_time):
            # Get values at 4 poloidal positions
            values = signals_reshaped[:, r_idx, t_idx]
            
            # Linear interpolation
            f = interp1d(z_original, values, kind='linear', fill_value='extrapolate')
            signals_interpolated[:, r_idx, t_idx] = f(z_target)
    
    # Reshape to (64, n_time)
    signals_8x8 = signals_interpolated.reshape(n_poloidal_out * n_radial, n_time)
    
    return signals_8x8


def prepare_windows_for_model(signals_8x8, window_size=128, stride=32):
    """
    Prepare sliding windows from 8×8 signals
    
    Args:
        signals_8x8: (64, n_time) array
        window_size: window size in samples (default 128 µs)
        stride: step size in samples (default 32 µs)
    
    Returns:
        windows: (n_windows, 8, 8, 128) torch tensor
    """
    n_time = signals_8x8.shape[1]
    n_windows = (n_time - window_size) // stride + 1
    
    windows = np.zeros((n_windows, 8, 8, window_size))
    
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        window_data = signals_8x8[:, start_idx:end_idx]
        
        # Reshape to 8×8 spatial grid
        windows[i] = window_data.reshape(8, 8, window_size)
    
    return torch.FloatTensor(windows)


# ==================== VISUALIZATION ====================
def visualize_results(boundary, rpos, zpos, selected_indices, radial_indices,
                     t, all_signals, signals_4x8, signals_8x8,
                     predictions, stride, elm_positions, sampling_rate):
    """Create comprehensive visualization"""
    print("Generating visualizations...")
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    t_ms = t * 1000
    rpos_m = rpos
    zpos_m = zpos
    
    # 1. Plasma boundary with channel selection
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(boundary[:, 0], boundary[:, 1], 'k-', linewidth=2.5, label='LCFS')
    ax1.scatter(rpos_m[:, 0], zpos_m[:, 0], c='lightgray', s=30, 
               marker='o', alpha=0.5, label='All channels (4×16)')
    
    # Color selected channels (red=inside, blue=outside)
    for idx in selected_indices:
        r_ch = rpos_m[idx, 0]
        z_ch = zpos_m[idx, 0]
        bdry_r_mid = boundary[np.abs(boundary[:, 1]) < 0.1, 0]
        if len(bdry_r_mid) > 0 and r_ch > np.max(bdry_r_mid):
            color = 'red'
        else:
            color = 'red'
        ax1.plot(r_ch, z_ch, 'o', color=color, markersize=10, alpha=0.7)
    
    ax1.set_xlabel('R [m]', fontsize=11)
    ax1.set_ylabel('Z [m]', fontsize=11)
    ax1.set_title('BES Channel Selection (Red=Inside, Blue=Outside)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. Zoomed view
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(boundary[:, 0], boundary[:, 1], 'k-', linewidth=2.5, label='LCFS')
    ax2.scatter(rpos_m[:, 0], zpos_m[:, 0], c='lightgray', s=30, 
               marker='o', alpha=0.5)

    for idx in selected_indices:
        r_ch = rpos_m[idx, 0]
        z_ch = zpos_m[idx, 0]
        bdry_r_mid = boundary[np.abs(boundary[:, 1]) < 0.1, 0]
        if len(bdry_r_mid) > 0 and r_ch > np.max(bdry_r_mid):
            color = 'red'
        else:
            color = 'red'
        ax2.plot(r_ch, z_ch, 's', color=color, markersize=12, 
                markeredgecolor='black', markeredgewidth=1.5)
    
    for i, rad_idx in enumerate(radial_indices):
        idx = rad_idx
        ax2.text(rpos_m[idx, 0], zpos_m[idx, 0] - 0.005, f'{i+1}',
                fontsize=8, ha='center', va='top', fontweight='bold')
    
    ax2.set_xlabel('R [m]', fontsize=11)
    ax2.set_ylabel('Z [m]', fontsize=11)
    ax2.set_title('Selected 4×8 Channels (Zoomed)', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    r_min, r_max = rpos_m[selected_indices, 0].min(), rpos_m[selected_indices, 0].max()
    z_min, z_max = zpos_m[selected_indices, 0].min(), zpos_m[selected_indices, 0].max()
    margin = 0.05
    ax2.set_xlim([r_min - margin, r_max + margin])
    ax2.set_ylim([z_min - margin, z_max + margin])
    
    # 3. 4×8 signal heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    downsample = 100
    im = ax3.imshow(signals_4x8[:, ::downsample], aspect='auto', cmap='RdBu_r',
                   interpolation='bilinear', extent=[t_ms[0], t_ms[-1], 31.5, -0.5])
    ax3.set_xlabel('Time [ms]', fontsize=11)
    ax3.set_ylabel('Channel Index (4×8)', fontsize=11)
    ax3.set_title('Selected 4×8 BES Signals', fontsize=12, fontweight='bold')
    
    for elm_time in elm_positions:
        ax3.axvline(elm_time * 1000, color='yellow', linestyle='--', linewidth=2)
    
    plt.colorbar(im, ax=ax3, label='Amplitude [a.u.]')
    
    # 4. Interpolation demonstration
    ax4 = fig.add_subplot(gs[1, 0])
    sample_time_idx = int(0.015 * sampling_rate)
    radial_col = 0
    
    signal_4_poloidal = signals_4x8[[0, 8, 16, 24], sample_time_idx]
    signal_8_poloidal = signals_8x8[[i*8 + radial_col for i in range(8)], sample_time_idx]
    
    z_4 = zpos_m[selected_indices[[0, 8, 16, 24]], 0]
    z_8 = np.linspace(z_4[0], z_4[-1], 8)
    
    ax4.plot(z_4, signal_4_poloidal, 'bo-', markersize=10, linewidth=2, label='Original 4 points')
    ax4.plot(z_8, signal_8_poloidal, 'r^--', markersize=8, linewidth=1.5, label='Interpolated 8 points')
    ax4.set_xlabel('Z position [m]', fontsize=11)
    ax4.set_ylabel('Signal Amplitude', fontsize=11)
    ax4.set_title(f'Poloidal Interpolation (t=15ms, R-col {radial_col})', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 8×8 signal heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.imshow(signals_8x8[:, ::downsample], aspect='auto', cmap='RdBu_r',
                   interpolation='bilinear', extent=[t_ms[0], t_ms[-1], 63.5, -0.5])
    ax5.set_xlabel('Time [ms]', fontsize=11)
    ax5.set_ylabel('Channel Index (8×8)', fontsize=11)
    ax5.set_title('Interpolated 8×8 Signals for CNN', fontsize=12, fontweight='bold')
    
    for elm_time in elm_positions:
        ax5.axvline(elm_time * 1000, color='yellow', linestyle='--', linewidth=2)
    
    plt.colorbar(im, ax=ax5, label='Amplitude [a.u.]')
    
    # 6. Single channel comparison
    ax6 = fig.add_subplot(gs[1, 2])
    ch_4x8 = 12
    ch_8x8 = 20
    
    ax6.plot(t_ms, signals_4x8[ch_4x8, :], 'b-', linewidth=0.8, alpha=0.7, label=f'4×8 Ch {ch_4x8}')
    ax6.plot(t_ms, signals_8x8[ch_8x8, :], 'r-', linewidth=0.8, alpha=0.7, label=f'8×8 Ch {ch_8x8} (interp)')
    
    for elm_time in elm_positions:
        ax6.axvline(elm_time * 1000, color='red', linestyle='--', alpha=0.5)
    
    ax6.set_xlabel('Time [ms]', fontsize=11)
    ax6.set_ylabel('Amplitude [a.u.]', fontsize=11)
    ax6.set_title('Original vs Interpolated Signal', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([t_ms[0], t_ms[-1]])
    
    # 7. CNN Predictions
    ax7 = fig.add_subplot(gs[2, :2])
    pred_times = (np.arange(len(predictions)) * stride) / sampling_rate * 1000
    
    ax7.plot(pred_times, predictions, 'b-', linewidth=1.5, label='CNN Predictions')
    ax7.axhline(PREDICTION_THRESHOLD, color='green', linestyle='--', linewidth=2, label=f'Threshold ({PREDICTION_THRESHOLD})')
    
    for elm_time in elm_positions:
        ax7.axvline(elm_time * 1000, color='red', linestyle='--', linewidth=2, alpha=0.6)
    
    ax7.set_xlabel('Time [ms]', fontsize=12)
    ax7.set_ylabel('ELM Onset Probability', fontsize=12)
    ax7.set_title('BESNN Predictions', fontsize=13, fontweight='bold')
    ax7.set_ylim([0, 1.1])
    ax7.legend(loc='upper right', fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([t_ms[0], t_ms[-1]])
    
    # 8. Statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    pred_binary = (predictions > PREDICTION_THRESHOLD).astype(int)
    elm_detected = []
    for elm_time in elm_positions:
        elm_time_ms = elm_time * 1000
        mask = (pred_times > elm_time_ms - 2) & (pred_times < elm_time_ms + 2)
        if np.any(pred_binary[mask] == 1):
            elm_detected.append(True)
        else:
            elm_detected.append(False)
    
    stats_text = f"""Prediction Statistics
━━━━━━━━━━━━━━━━━━━━

Configuration:
• Total channels: {len(rpos)}
• Selected: {len(selected_indices)} (4×8)
• Interpolated: 64 (8×8)

Predictions:
• Windows: {len(predictions)}
• Window: {WINDOW_SIZE} µs
• Stride: {stride} µs
• Threshold: {PREDICTION_THRESHOLD}
• Range: [{predictions.min():.3f}, {predictions.max():.3f}]
• ELMs detected: {sum(elm_detected)}/{len(elm_positions)}

Files:
• HDF5: {os.path.basename(HDF5_FILE)}
• Model: {os.path.basename(MODEL_FILE)}
"""
    
    ax8.text(0.05, 0.95, stats_text, fontsize=9.5, verticalalignment='top',
            fontfamily='monospace', transform=ax8.transAxes,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', 
                     alpha=0.9, edgecolor='black', linewidth=2))
    
    # 9. Workflow
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')
    
    workflow_text = """Workflow Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Load HDF5 → Geometry (boundary, BES positions) + Signals (64 channels)
2. Calculate 4×8 selection → Based on boundary crossing
3. Extract 4×8 signals → 32 channels from 64 total
4. Interpolate 4×8 → 8×8 → Linear interpolation along poloidal (Z) direction
5. Normalize & Window → Per-µs normalization, 128 µs windows, 32 µs stride
6. Load pre-trained model → BESNN CNN from .pt file
7. Run predictions → ELM onset probabilities [0, 1]
8. Visualize results

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    ax9.text(0.05, 0.95, workflow_text, fontsize=9, verticalalignment='top',
            fontfamily='monospace', transform=ax9.transAxes,
            bbox=dict(boxstyle='round,pad=1', facecolor='wheat', 
                     alpha=0.9, edgecolor='black', linewidth=2))
    
    plt.suptitle('BESNN: ELM Onset Prediction Using 4×16 BES Configuration',
                fontsize=16, fontweight='bold', y=0.998)
    
    output_path = 'besnn_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Figure saved: {output_path}\n")
    plt.close()


# ==================== MAIN ====================
def main():
    """Main execution function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load HDF5 data
    (boundary, rpos, zpos, t, all_signals, elm_positions,
     selected_indices_saved, radial_indices_saved, sampling_rate) = load_from_hdf5(HDF5_FILE)
    
    # Calculate channel selection from boundary (ALWAYS recalculate)
    n_radial = len(rpos) // (len(rpos) // 16) if len(rpos) == 64 else 16
    selected_indices, radial_indices = select_4x8_channels(boundary, rpos, zpos, n_radial)
    
    # Validate against saved selection if available
    if selected_indices_saved is not None:
        print("="*70)
        print("VALIDATING CALCULATED SELECTION AGAINST SAVED")
        print("="*70)
        if np.array_equal(selected_indices, selected_indices_saved):
            print("✓ VALIDATION PASSED: Calculated selection matches saved selection")
        else:
            print("⚠ VALIDATION WARNING: Calculated selection differs from saved")
            print(f"  Calculated: {len(selected_indices)} channels")
            print(f"  Saved: {len(selected_indices_saved)} channels")
            diff = len(set(selected_indices) ^ set(selected_indices_saved))
            print(f"  Difference: {diff} channels")
        print("="*70 + "\n")
    
    # Extract 4×8 signals
    signals_4x8 = all_signals[selected_indices, :]
    print(f"Extracted 4×8 signals: {signals_4x8.shape}")
    
    # Interpolate to 8×8
    print("Interpolating 4×8 → 8×8...")
    signals_8x8 = interpolate_4x8_to_8x8(signals_4x8)
    print(f"Interpolated to 8×8: {signals_8x8.shape}")
    
    # Normalize signals (per-microsecond normalization)
    print("Normalizing signals...")
    normalized_signals = np.zeros_like(signals_8x8)
    for t_idx in range(signals_8x8.shape[1]):
        std = np.std(signals_8x8[:, t_idx])
        if std > 1e-10:
            normalized_signals[:, t_idx] = signals_8x8[:, t_idx] / std
        else:
            normalized_signals[:, t_idx] = signals_8x8[:, t_idx]
    
    # Prepare windows for CNN
    print("Preparing windows for model...")
    windows = prepare_windows_for_model(normalized_signals, window_size=WINDOW_SIZE, stride=STRIDE)
    print(f"Created {len(windows)} windows")
    
    # Load model
    model, checkpoint = load_model(MODEL_FILE, device)
    
    # Run predictions
    print("Running predictions...")
    model.eval()
    with torch.no_grad():
        windows_gpu = windows.to(device)
        predictions = model(windows_gpu).cpu().numpy().flatten()
    
    print(f"✓ Predictions complete")
    print(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]\n")
    
    # Visualize results
    visualize_results(boundary, rpos, zpos, selected_indices, radial_indices,
                     t, all_signals, signals_4x8, signals_8x8,
                     predictions, STRIDE, elm_positions, sampling_rate)
    
    print("="*70)
    print("EXECUTION COMPLETE!")
    print("="*70)
    print(f"Output: besnn_predictions.png")
    print("="*70)


if __name__ == "__main__":
    main()
