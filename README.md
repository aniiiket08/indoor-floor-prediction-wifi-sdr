<div align="center">

# Indoor Floor Prediction

**WiFi fingerprinting with a 1D CNN, and real RF capture on a USRP B210.**

![Python](https://img.shields.io/badge/python-3.10-3b4252?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-1D%20CNN-3b4252?style=flat-square&logo=pytorch&logoColor=white)
![GNU Radio](https://img.shields.io/badge/gnu%20radio-gr--ieee802--11-3b4252?style=flat-square)
![USRP](https://img.shields.io/badge/ettus-usrp%20b210-3b4252?style=flat-square)

[Overview](#overview) · [Status](#status) · [Pipeline](#pipeline) · [Getting started](#getting-started) · [Usage](#usage) · [Structure](#project-structure)

</div>

<br>

## Overview

GPS does not work reliably indoors, since walls and ceilings degrade the satellite signal. WiFi fingerprinting gets around this by treating the signal strengths of nearby access points as a location signature.

This project trains a 1D CNN to predict which floor of a building a device is on, and pairs it with a software-defined radio to capture real 2.4 GHz signals.

|  |  |
|---|---|
| **Dataset** | [UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) |
| **Input** | 520 access-point RSSI values |
| **Output** | Floor label (0 to 3) |
| **Model** | 1D CNN in PyTorch |
| **Hardware** | Ettus USRP B210 over USB 3.0 |
| **Decoding** | GNU Radio with `gr-ieee802-11` |

<br>

## Status

| Component | State |
|---|---|
| Preprocessing, training, prediction on UJIIndoorLoc | Complete |
| USRP B210 bring-up (UHD, GNU Radio, udev rules) | Complete |
| Raw IQ capture and power estimation | Complete |
| Floor prediction from live local captures | Blocked, see below |

> [!IMPORTANT]
> **The model cannot yet predict floors for the local building.**
> Its 520 input columns are anonymized access points recorded at Jaume I University, Spain. The B210 decodes real local MAC addresses and RSSI values, but none of them correspond to those columns.

Two ways forward:

1. **Retrain on local data (recommended).** Collect labeled RSSI fingerprints per floor in the target building and retrain the CNN on them.
2. **Site-survey mapping.** Record a MAC to WAP index mapping in `wap_mapping.json`. This confirms the pipeline is wired correctly, but the model still will not recognize this building.

<br>

## Pipeline

<p align="center">
  <img src="assets/pipeline_diagram.png" alt="RF environment to USRP B210 to raw IQ to gr-ieee802-11 decoding to fingerprint extraction to CNN to predicted floor" width="860">
</p>

<br>

## Getting started

**Requirements**

- Ubuntu 22.04 LTS
- Python 3.10
- Ettus USRP B210 on a USB 3.0 port

<details>
<summary><b>Install the SDR stack</b></summary>

<br>

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

</details>

**Set up the Python environment**

```bash
cd uji_floor_predict
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Check the radio is detected**

```bash
uhd_find_devices
```

<br>

## Usage

### 1. Train and evaluate

```bash
python3 src/train.py
python3 src/predict.py
```

The best checkpoint is saved to `models/best_model.pth`.

### 2. Capture IQ samples

```bash
uhd_rx_cfile --freq 2.4e9 --rate 20e6 --gain 30 --nsamps 1000000 -o captures/location1.dat
```

### 3. Check the capture

```bash
python3 plot_capture.py --dat captures/location1.dat \
    --rate 20e6 --freq 2.4e9 --out assets/capture_location1.png
```

Plots received power over time and the spectrum around the tuned frequency. A flat, near-zero plot usually means a disconnected antenna, low gain, or nothing transmitting in band.

### 4. Decode beacon frames

Run the `gr-ieee802-11` flowgraph in GNU Radio Companion and log `(mac, rssi_dbm)` for each decoded frame to a CSV.

### 5. Build a fingerprint

```bash
python3 extract_fingerprint.py \
    --dat captures/location1.dat \
    --packets captures/location1_decoded.csv \
    --mapping wap_mapping.json \
    --out captures/fingerprint.csv
```

<br>

## Project structure

```
.
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── models/
│   └── best_model.pth
├── captures/                raw .dat files, decoded packet CSVs
├── assets/                  diagrams, capture plots
├── extract_fingerprint.py   capture to 520-dim fingerprint
├── plot_capture.py          capture diagnostics
├── wap_mapping.json         MAC to WAP mapping (optional)
├── requirements.txt
└── README.md
```

<br>

## Glossary

<details>
<summary><b>Terms used in this project</b></summary>

<br>

| Term | Meaning |
|---|---|
| RSSI | Received signal strength in dBm. In UJIIndoorLoc, -105 means not detected. |
| WAP | WiFi access point. One feature column per AP. |
| SDR | Software-defined radio. RF processing done in software, so it can be retuned across frequencies and protocols. |
| IQ samples | Raw SDR output (in-phase and quadrature) that together encode amplitude and phase. |
| UHD | Ettus's USRP Hardware Driver. |
| `gr-ieee802-11` | GNU Radio module that decodes 802.11 frames from IQ samples. |

</details>

<br>

## Acknowledgements

Training data from [UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc), UCI Machine Learning Repository. Frame decoding by [gr-ieee802-11](https://github.com/bastibl/gr-ieee802-11).
