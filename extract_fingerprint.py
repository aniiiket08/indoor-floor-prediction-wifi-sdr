"""
extract_fingerprint.py

Turns a real USRP B210 capture into a 520-dim WiFi RSSI fingerprint vector
in the same format as the UJIIndoorLoc dataset (WAP001..WAP520, -105 = not
detected), so it can be fed straight into predict_floor().

HONEST SCOPE / KNOWN LIMITATION — READ THIS BEFORE USING
----------------------------------------------------------------
This script does two genuinely real things:

1. Reads the raw IQ samples from a .dat file captured with `uhd_rx_cfile`
   and computes actual signal power / RSSI from them. This part is real
   DSP, no shortcuts.

2. Builds a 520-dim fingerprint vector from *decoded WiFi packets*
   (MAC address + per-packet RSSI), which you get by running the
   gr-ieee802-11 flowgraph against the B210 and logging its output.
   Demodulating raw IQ into decoded 802.11 beacon frames from scratch
   is a full PHY-layer receiver (correlation, OFDM demod, Viterbi
   decoding, CRC check) — that is exactly what gr-ieee802-11 already
   implements, so this script does not try to reinvent it in pure
   Python. It expects gr-ieee802-11's decoded output as input.

THE PART THAT CANNOT BE FAKED OR ENGINEERED AROUND:
UJIIndoorLoc's WAP001..WAP520 columns are anonymized MAC addresses that
were captured at Jaume I University, Spain. They do not exist in your
building. There is no legitimate way to map an access point you detect
locally onto one of those 520 specific columns unless you built your
OWN mapping — i.e. you personally walked your building, recorded which
physical AP corresponds to which WAP index, and saved that mapping to
wap_mapping.json. Without that survey, every WAP column in the output
vector is correctly reported as -105 (not detected), because from the
model's point of view, none of *your* real APs are anything it was
trained to recognize. That's not a limitation of this script — it's a
real mismatch between your deployment environment and the training
dataset, and it should be disclosed as such in your writeup.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

NUM_WAPS = 520
NOT_DETECTED = -105
WAP_COLUMNS = [f"WAP{i:03d}" for i in range(1, NUM_WAPS + 1)]


# ---------------------------------------------------------------------
# 1. Real signal processing on the raw .dat capture
# ---------------------------------------------------------------------
def read_iq_samples(dat_path: str, dtype: str = "complex64") -> np.ndarray:
    """
    Reads raw IQ samples from a .dat file produced by:
        uhd_rx_cfile --freq 2.4e9 --rate <rate> --gain <gain> -o file.dat

    uhd_rx_cfile writes interleaved 32-bit float I/Q pairs by default,
    which numpy can read directly as complex64.
    """
    path = Path(dat_path)
    if not path.exists():
        raise FileNotFoundError(f"Capture file not found: {dat_path}")

    samples = np.fromfile(path, dtype=np.complex64 if dtype == "complex64" else dtype)
    if samples.size == 0:
        raise ValueError(f"{dat_path} contains no samples — capture may have failed")
    return samples


def estimate_capture_power(samples: np.ndarray) -> dict:
    """
    Computes real signal statistics from the IQ samples:
      - mean power (linear, in the SDR's raw ADC units)
      - power in dB (relative — NOT calibrated absolute dBm, since that
        requires knowing the receiver's full gain chain / cable loss,
        which uhd_rx_cfile does not record)
      - peak power, useful for sanity-checking that the capture isn't
        empty / all-zero (a common failure mode if the antenna wasn't
        connected or the gain was 0)
    """
    power = np.abs(samples) ** 2  # power = |I + jQ|^2
    mean_power = float(np.mean(power))
    peak_power = float(np.max(power))

    # Convert to relative dB. This is NOT calibrated dBm — treat it as
    # a relative signal-strength indicator for sanity checks, not as
    # ground truth RSSI for the fingerprint.
    mean_power_db = 10 * np.log10(mean_power + 1e-12)
    peak_power_db = 10 * np.log10(peak_power + 1e-12)

    return {
        "num_samples": int(samples.size),
        "mean_power_linear": mean_power,
        "peak_power_linear": peak_power,
        "mean_power_db": round(mean_power_db, 2),
        "peak_power_db": round(peak_power_db, 2),
    }


# ---------------------------------------------------------------------
# 2. Building the 520-dim fingerprint from decoded WiFi packets
# ---------------------------------------------------------------------
def load_decoded_packets(packets_csv: str) -> list[dict]:
    """
    Loads decoded WiFi beacon/probe packets logged from the
    gr-ieee802-11 flowgraph. Expected CSV columns: mac,rssi_dbm

    You get this file by running gr-ieee802-11's wifi_phy_hier +
    wifi_mac blocks against the B210 in GNU Radio Companion and adding
    a message sink that writes (mac, rssi) per decoded frame to CSV.
    This script does not decode 802.11 itself.
    """
    path = Path(packets_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"Decoded packets file not found: {packets_csv}\n"
            "This must come from gr-ieee802-11's decoded output, not "
            "generated by hand."
        )

    packets = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                packets.append({
                    "mac": row["mac"].strip().lower(),
                    "rssi_dbm": float(row["rssi_dbm"]),
                })
            except (KeyError, ValueError) as e:
                print(f"Skipping malformed row {row}: {e}", file=sys.stderr)
    return packets


def load_wap_mapping(mapping_path: str) -> dict:
    """
    Loads your own MAC -> WAP index mapping, built from a manual site
    survey of your building. Format:
        {
          "aa:bb:cc:dd:ee:ff": 12,
          "11:22:33:44:55:66": 87
        }
    If this file is empty or missing entries for a detected MAC, that
    WAP stays -105 in the output — which is the honest result, since
    the model has no column for an AP it never saw during training.
    """
    path = Path(mapping_path)
    if not path.exists():
        print(
            f"No WAP mapping file found at {mapping_path} — every "
            "detected AP will be reported as -105 (unmapped). See the "
            "module docstring for why this mapping can't be generated "
            "automatically.",
            file=sys.stderr,
        )
        return {}
    with open(path) as f:
        return json.load(f)


def build_fingerprint(packets: list[dict], wap_mapping: dict) -> np.ndarray:
    """
    Produces a 520-dim vector in WAP001..WAP520 order.
    Detected + mapped APs get their real measured RSSI.
    Everything else is -105 (not detected), matching UJIIndoorLoc's
    own convention for missing WAPs.
    """
    vector = np.full(NUM_WAPS, NOT_DETECTED, dtype=float)

    matched, unmatched = 0, 0
    for pkt in packets:
        wap_index = wap_mapping.get(pkt["mac"])
        if wap_index is None:
            unmatched += 1
            continue
        if not (1 <= wap_index <= NUM_WAPS):
            print(f"WAP index {wap_index} for {pkt['mac']} out of range, skipping",
                  file=sys.stderr)
            continue
        # If the same AP was heard multiple times, keep the strongest reading
        vector[wap_index - 1] = max(vector[wap_index - 1], pkt["rssi_dbm"])
        matched += 1

    print(f"Matched {matched} detected APs to known WAP columns.")
    print(f"{unmatched} detected APs had no entry in wap_mapping.json "
          f"(reported as -105).")
    return vector


def save_fingerprint_csv(vector: np.ndarray, out_path: str, floor_label: int | None = None):
    row = {col: val for col, val in zip(WAP_COLUMNS, vector)}
    if floor_label is not None:
        row["FLOOR"] = floor_label

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"Fingerprint written to {out_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build a real 520-dim RSSI fingerprint from a B210 capture "
                    "and gr-ieee802-11 decoded output."
    )
    parser.add_argument("--dat", required=True,
                        help="Path to the raw .dat IQ capture (for power sanity check)")
    parser.add_argument("--packets", required=True,
                        help="CSV of decoded packets (mac,rssi_dbm) from gr-ieee802-11")
    parser.add_argument("--mapping", default="wap_mapping.json",
                        help="Your own MAC->WAP index mapping (see docstring)")
    parser.add_argument("--out", default="captures/fingerprint.csv",
                        help="Where to write the resulting 520-dim CSV row")
    parser.add_argument("--floor-label", type=int, default=None,
                        help="Optional ground-truth floor label to store alongside the row")
    args = parser.parse_args()

    # Step 1: real sanity check on the raw capture
    samples = read_iq_samples(args.dat)
    stats = estimate_capture_power(samples)
    print("Capture stats:", stats)
    if stats["peak_power_db"] < -60:
        print("WARNING: capture looks empty/very weak — check antenna, "
              "gain, and that the B210 was actually receiving on 2.4 GHz.",
              file=sys.stderr)

    # Step 2: build the fingerprint from decoded packets + your mapping
    packets = load_decoded_packets(args.packets)
    wap_mapping = load_wap_mapping(args.mapping)
    fingerprint = build_fingerprint(packets, wap_mapping)

    save_fingerprint_csv(fingerprint, args.out, args.floor_label)


if __name__ == "__main__":
    main()
