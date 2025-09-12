#!/usr/bin/env python3
"""
Create normalized TSpline3 from feather file for SiPM simulation

Usage: python scripts/make_spline.py
"""

import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# File paths
feather_file = "../SPR_data/Sensl_FastOut_AveragePulse_1p8GHzBandwidth.feather"
original_file = "/fs/ddn/sdf/group/atlas/d/liangyu/dSiPM/fasttiming/SPR_data/SiPM_g2_pulse.root"
output_file = "../SPR_data/SiPM_TTU_pulse.root"

# Auto-determine normalization parameters
print("Determining normalization parameters...")

# Load original pulse
try:
    fin = ROOT.TFile.Open(original_file)
    spline_orig = fin.Get("pulseShapeSpline")
    t_min, t_max = spline_orig.GetXmin(), spline_orig.GetXmax()
    time_orig = np.arange(t_min, t_max + 0.04, 0.04)
    amp_orig = np.array([spline_orig.Eval(t) for t in time_orig])
    orig_baseline = np.mean(amp_orig[:len(amp_orig)//10])
    orig_signal = amp_orig.max() - orig_baseline
    fin.Close()
    print(f"Original: peak={amp_orig.max():.3f}, baseline={orig_baseline:.3f}, signal={orig_signal:.3f}")
except:
    print("Using default original pulse parameters")
    orig_signal = 0.85

# Load feather file and extract window
df = pd.read_feather(feather_file)
peak_region = df[(df['time'] >= 30) & (df['time'] <= 70)].copy()
time_raw = peak_region['time'].values
amplitude_raw = peak_region['data_ch0'].values

# Auto-calculate normalization parameters
peak_idx = np.argmax(amplitude_raw)
peak_time = time_raw[peak_idx]
baseline_subtract = np.mean(amplitude_raw[:len(amplitude_raw)//10])
new_signal = amplitude_raw.max() - baseline_subtract
scale_factor = orig_signal / new_signal

print(f"New pulse: peak={amplitude_raw.max():.0f}, baseline={baseline_subtract:.1f}, signal={new_signal:.0f}")
print(f"Normalization: baseline_subtract={baseline_subtract:.1f}, scale_factor={scale_factor:.6f}")

# Apply normalization
time_normalized = time_raw - peak_time
amplitude_normalized = (amplitude_raw - baseline_subtract) * scale_factor

# Create TSpline3
fout = ROOT.TFile.Open(output_file, "RECREATE")

# Convert to proper ROOT arrays
n_points = len(time_normalized)
time_array = ROOT.TArrayD(n_points, time_normalized.astype(np.float64))
amp_array = ROOT.TArrayD(n_points, amplitude_normalized.astype(np.float64))

# Create TSpline3 with ROOT arrays
spline = ROOT.TSpline3("pulseShapeSpline", time_array.GetArray(), amp_array.GetArray(), n_points)
spline.SetTitle("SiPM pulse - 1.8GHz, 25GHz sampling")
spline.SetName("pulseShapeSpline")  # Ensure the name is set correctly

# Write and close
spline.Write()
fout.Write()  # Ensure everything is written
fout.Close()


# Generate overlay comparison plot
try:
    fin_new = ROOT.TFile.Open(output_file, "READ")
    spline_new = fin_new.Get("Spline3;1") or fin_new.Get("pulseShapeSpline;1")
    
    if spline_new:
        t_min_new, t_max_new = spline_new.GetXmin(), spline_new.GetXmax()
        time_new_verify = np.arange(t_min_new, t_max_new + 0.04, 0.04)
        amp_new_verify = np.array([spline_new.Eval(t) for t in time_new_verify])
        
        plt.figure(figsize=(10, 6))
        if 'amp_orig' in locals():
            plt.plot(time_orig, amp_orig, 'b-', linewidth=2, label='Original SiPM_g2_pulse')
        plt.plot(time_new_verify, amp_new_verify, 'r-', linewidth=2, label='New SiPM_TTU_pulse')
        plt.xlabel('Time (ns)')
        plt.ylabel('Amplitude')
        plt.title('SiPM Pulse Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-25, 25)
        plt.tight_layout()
        plt.savefig('../SPR_data/pulse_overlay.png', dpi=150, bbox_inches='tight')
        print("Overlay plot saved: ../SPR_data/pulse_overlay.png")
    fin_new.Close()
except:
    print("Could not generate overlay plot")

print(f"Created TSpline3: {output_file}")
print(f"Time range: {time_normalized.min():.1f} to {time_normalized.max():.1f} ns")
print(f"Amplitude range: {amplitude_normalized.min():.3f} to {amplitude_normalized.max():.3f}")
