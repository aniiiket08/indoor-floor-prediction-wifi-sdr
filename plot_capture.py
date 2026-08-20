"""
plot_capture.py

Generates real diagnostic plots from an actual USRP B210 .dat capture:
  1. Instantaneous power over time (|I + jQ|^2 per sample)
  2. Power spectral density (FFT-based), showing energy across the
     captured bandwidth around your tuned center frequency

Use this to visually confirm a capture actually picked up RF energy
(a flat near-zero power plot means the antenna wasn't connected, the
gain was too low, or nothing was transmitting in-band) and to include
a genuine data plot in your report/README instead of a diagram alone.

Usage:
    python3 plot_capture.py --dat captures/location1.dat \
        --rate 20e6 --freq 2.4e9 --out assets/capture_location1.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def read_iq_samples(dat_path: str) -> np.ndarray:
    path = Path(dat_path)
    if not path.exists():
        raise FileNotFoundError(f"Capture file not found: {dat_path}")
    samples = np.fromfile(path, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"{dat_path} contains no samples")
    return samples


def plot_capture(samples: np.ndarray, sample_rate: float, center_freq: float,
                  out_path: str, max_samples_time: int = 200_000):
    power = np.abs(samples) ** 2
    power_db = 10 * np.log10(power + 1e-12)

    # Time-domain view (subsampled for a readable plot on long captures)
    n_show = min(len(power_db), max_samples_time)
    t = np.arange(n_show) / sample_rate * 1e3  # ms

    # PSD via Welch-style averaged FFT
    nfft = 4096
    n_segments = max(1, len(samples) // nfft)
    psd_acc = np.zeros(nfft)
    for i in range(n_segments):
        seg = samples[i * nfft:(i + 1) * nfft]
        if len(seg) < nfft:
            break
        windowed = seg * np.hanning(nfft)
        spectrum = np.fft.fftshift(np.fft.fft(windowed))
        psd_acc += np.abs(spectrum) ** 2
    psd_db = 10 * np.log10(psd_acc / max(n_segments, 1) + 1e-12)
    freqs_mhz = (np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / sample_rate)) + center_freq) / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200)

    axes[0].plot(t, power_db[:n_show], linewidth=0.5, color="#2b6cb0")
    axes[0].set_title("Received Power vs Time")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Power (dB, relative)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(freqs_mhz, psd_db, linewidth=0.8, color="#c53030")
    axes[1].set_title(f"Power Spectral Density (center {center_freq/1e6:.1f} MHz)")
    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylabel("Power (dB, relative)")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"B210 Capture Diagnostics — {len(samples):,} samples "
                 f"@ {sample_rate/1e6:.1f} Msps", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to {out}")

    peak_db = float(np.max(power_db))
    mean_db = float(np.mean(power_db))
    print(f"Mean power: {mean_db:.2f} dB (relative)  |  Peak power: {peak_db:.2f} dB (relative)")
    if peak_db < -60:
        print("WARNING: capture looks empty/very weak — check antenna, gain, "
              "and that something was actually transmitting on this frequency.")


def main():
    parser = argparse.ArgumentParser(description="Plot real power/spectrum from a B210 .dat capture")
    parser.add_argument("--dat", required=True, help="Path to the .dat IQ capture")
    parser.add_argument("--rate", type=float, required=True, help="Sample rate used during capture (Hz)")
    parser.add_argument("--freq", type=float, required=True, help="Center frequency used during capture (Hz)")
    parser.add_argument("--out", default="assets/capture_diagnostics.png", help="Output PNG path")
    args = parser.parse_args()

    samples = read_iq_samples(args.dat)
    plot_capture(samples, args.rate, args.freq, args.out)


if __name__ == "__main__":
    main()
