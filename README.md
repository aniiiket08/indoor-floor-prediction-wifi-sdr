# Indoor Floor Prediction

Floor-level indoor localization from WiFi RSSI fingerprints. A 1D CNN is trained on the UJIIndoorLoc dataset, and a USRP B210 software-defined radio is used to capture real 2.4 GHz signals.

GPS is unreliable indoors because walls and ceilings attenuate satellite signals. WiFi fingerprinting avoids this by treating the signal strengths from nearby access points as a location signature.

## Status

| Component | State |
|---|---|
| Model training and prediction (UJIIndoorLoc) | Working |
| USRP B210 setup (UHD, GNU Radio, `gr-ieee802-11`) | Working |
| Raw IQ capture and power estimation | Working |
| Live floor prediction from local captures | Not yet possible, see below |

### Known limitation

The 520 input columns of the trained model are anonymized access points recorded at Jaume I University, Spain. They do not correspond to any access point in the building where the SDR captures are made. Decoding beacons with the B210 yields real local MAC addresses and RSSI values, but there is no way to map them onto the dataset's `WAP001`–`WAP520` columns in a meaningful way.

Two ways forward:

1. **Retrain on local data (recommended).** Collect labeled RSSI fingerprints per floor in the target building and retrain the CNN on them.
2. **Site-survey mapping.** Record a MAC to WAP index mapping in `wap_mapping.json`. This verifies that the pipeline is wired correctly, but the model still cannot recognize this building.

## Pipeline

![Pipeline](assets/pipeline_diagram.png)

RF environment → USRP B210 → raw IQ → `gr-ieee802-11` decoding → fingerprint extraction → 1D CNN → predicted floor

## Requirements

- Ubuntu 22.04 LTS
- Python 3.10
- PyTorch (see `requirements.txt`)
- UHD and GNU Radio
- [`gr-ieee802-11`](https://github.com/bastibl/gr-ieee802-11)
- Ettus USRP B210 on a USB 3.0 port

## Setup

Install the SDR stack:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential git curl wget python3-pip python3-venv -y
sudo apt install libuhd-dev uhd-host -y
sudo uhd_images_downloader
sudo apt install gnuradio -y
sudo apt install cmake libboost-all-dev liblog4cpp5-dev swig -y

git clone https://github.com/bastibl/gr-ieee802-11.git
cd gr-ieee802-11 && mkdir build && cd build
cmake .. && make -j4 && sudo make install && sudo ldconfig
```

Set up the Python environment:

```bash
cd uji_floor_predict
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Check that the radio is detected:

```bash
uhd_find_devices
```

## Usage

**Train and evaluate on UJIIndoorLoc**

```bash
python3 src/train.py
python3 src/predict.py
```

The trained checkpoint is saved to `models/best_model.pth`.

**Capture raw IQ samples**

```bash
uhd_rx_cfile --freq 2.4e9 --rate 20e6 --gain 30 --nsamps 1000000 -o captures/location1.dat
```

**Decode beacon frames**

Run the `gr-ieee802-11` flowgraph in GNU Radio Companion and log `(mac, rssi_dbm)` for each decoded frame to a CSV.

**Check a capture**

```bash
python3 plot_capture.py --dat captures/location1.dat \
    --rate 20e6 --freq 2.4e9 --out assets/capture_location1.png
```

Plots received power over time and the spectrum around the center frequency. A flat, near-zero plot usually means a disconnected antenna, low gain, or nothing transmitting in band.

**Build a fingerprint from a capture**

```bash
python3 extract_fingerprint.py \
    --dat captures/location1.dat \
    --packets captures/location1_decoded.csv \
    --mapping wap_mapping.json \
    --out captures/fingerprint.csv
```

## Project layout

```
.
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── models/
│   └── best_model.pth
├── captures/               raw .dat files and decoded packet CSVs
├── assets/                 diagrams and capture plots
├── extract_fingerprint.py  capture to 520-dim fingerprint
├── plot_capture.py         capture diagnostics
├── wap_mapping.json        MAC to WAP mapping (optional)
├── requirements.txt
└── README.md
```

## Terms

| Term | Meaning |
|---|---|
| RSSI | Received signal strength in dBm. In UJIIndoorLoc, -105 means not detected. |
| WAP | WiFi access point. One feature column per AP. |
| IQ samples | Raw SDR output (in-phase and quadrature) encoding amplitude and phase. |
| UHD | Ettus's USRP Hardware Driver. |
| `gr-ieee802-11` | GNU Radio module that decodes 802.11 frames from IQ samples. |

## Data

[UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc), UCI Machine Learning Repository.
