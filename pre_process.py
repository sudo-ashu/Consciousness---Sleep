import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly, iirnotch, filtfilt, welch
import os
import random

fs_old = 1000
fs_new = 200
bin_sec = 2
bin_len = fs_new * bin_sec

def load_session(folder):
    signals = []
    for ch in range(1, 129):
        mat = sio.loadmat(os.path.join(folder, f"ECoG_ch{ch}.mat"))
        keys = [k for k in mat.keys() if not k.startswith('__')]
        sig = mat[keys[0]].squeeze()
        signals.append(sig)
    return np.vstack(signals)  # (128, T)

def preprocess(data):
    # downsampling
    random.seed(42)
    data_ds = resample_poly(data, up=1, down=5, axis=1).astype(np.float32)
    #average re-reference (demeaning across electrodes)
    data_mean = data_ds - np.mean(data_ds, axis=0, keepdims=True)
    # 50 Hz notch filter
    fs = 200
    f0 = 50
    Q = 30
    # recursive filter (infinite impluse response)
    b, a = iirnotch(f0, Q, fs) 
    data_filt = filtfilt(b, a, data_mean, axis=1) # filtfilt is used for zero-phase digital filtering...

    return data_ds, data_mean, data_filt

def make_bins(data, fs=200, bin_sec=2):
    bin_len = fs * bin_sec
    total_samples = data.shape[1]

    n_bins = total_samples // bin_len
    print(f"total_samples= {total_samples}")
    print(f"bins = {n_bins}")
    bins = []

    
    for i in range(n_bins):
        start = i * bin_len
        end = start + bin_len
        bins.append(data[:, start:end])
    return bins

def pick_random_bins(bins, n=20):
    return random.sample(bins, n)

experiment_info = {
    "Session1": {
        "Sleep": [
            (37.94, 2670.57)
        ]
    },

    "Session2": {
        "Sleep": [
            (9.24, 1550.78)
        ],

        "Awake": [
            (1600.13, 2048.28)
        ]
    },

    "Session3": {
        "Awake": [
            (0.83, 658.49)
        ]
    }
}

Regions = {
    "MP": [52],
    "LP": [25],
    "PM": [39],
    "MS": [30],
    "PC": [11],
    "TC": [94],
    "HV": [85],
    "LC": [102]
}

def main_function(session_folders):
    state_bins = {}
    # works on the meta data
    
    for folder in session_folders:
        session_name = os.path.basename(folder)
        print(session_name)
        
        raw = load_session(folder)
        data_ds, data_ref, data_filt = preprocess(raw)

        session_info = experiment_info[session_name]

        for state, intervals in session_info.items():
            if state not in state_bins:
                state_bins[state] = []

            for start_time, end_time in intervals:
                start_idx = int(start_time * fs_new)
                end_idx = int(end_time * fs_new)

                segment = data_filt[:, start_idx:end_idx]
                bins = make_bins(segment)
                state_bins[state].extend(bins)

    state_bins.keys()

session_folders = ["KT_George_Day_1/Session1", "KT_George_Day_1/Session2", "KT_George_Day_1/Session3"]
state_bins = main_function(session_folders)

