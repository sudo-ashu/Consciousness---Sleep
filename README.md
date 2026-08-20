# ECoG Consciousness-State Preprocessing

This project prepares multichannel electrocorticography (ECoG) recordings from a monkey experiment for comparisons across consciousness states. It converts raw, per-electrode MATLAB files into consistently preprocessed, region-specific recordings that can be used in later analysis.

The current workflow uses data from `KT_George_Day_2`, with:

- `Session1`: awake intervals
- `Session2`: anesthetized interval

An additional recovery-state interval is defined for `Session3` in the code, but that session is not included in the current processing run.

## Input data

Each session folder must contain one MATLAB file per electrode, named:

```text
ECoG_ch1.mat
ECoG_ch2.mat
...
ECoG_ch128.mat
```

Each file is expected to contain one ECoG time-series variable. The loader reads the first non-metadata MATLAB variable from every file and stacks the signals into an array of shape `(128, samples)`.

## Preprocessing modus operandi

The preprocessing implemented in `pre_process.py` is applied independently to every session, in this order:

1. **Downsample:** signals sampled at 1,000 Hz are resampled to 200 Hz using polyphase resampling (`resample_poly`, downsampling factor 5). This reduces data volume while retaining frequencies below the new 100 Hz Nyquist limit.
2. **Common-average reference:** at every time point, the mean across all 128 electrodes is subtracted from each channel. This reduces activity common to the electrode array.
3. **Power-line suppression:** a 50 Hz IIR notch filter with quality factor `Q = 30` is applied using forward-backward filtering (`filtfilt`). Forward-backward filtering produces zero phase shift in the processed signal.
4. **Region selection:** the preprocessed channels are subset to one of the predefined regions of interest (ROIs). Channel numbers in the ROI definition are one-based and are converted to Python's zero-based indexing.
5. **State segmentation:** each ROI signal is cropped to manually defined start/end times for the experimental state. Times are converted to sample indices using the 200 Hz post-downsampling rate.
6. **Epoching:** each state segment is divided into non-overlapping 2-second bins. Every bin therefore has shape `(number of ROI channels, 400 samples)`. Any incomplete final bin is discarded.

The state intervals currently encoded in `pre_process.py` are:

| Session | State | Interval(s), seconds |
| --- | --- | --- |
| `Session1` | Awake | 0.77–810.58; 2553.92–3301.07 |
| `Session2` | Anesthetized | 365.46–964.20 |
| `Session3` (configured, not run) | Recovery_Open | 28.76–901.93 |

## Regions of interest

`main_process.py` creates bins for the whole brain (`WB`, all 128 channels) and eight anatomical ROIs: `MP`, `LP`, `PM`, `MS`, `PC`, `TC`, `HV`, and `LC`. The exact electrode assignments are defined in the `ROI` dictionary in that file.

## Running the workflow

From this folder, run:

```bash
python main_process.py
python make_recordings.py
```

`main_process.py` loads the configured sessions and builds the in-memory dictionary:

```text
region_state_bins[region][state] -> list of 2-second arrays
```

`make_recordings.py` imports that result, deterministically samples 150 bins per region/state (seed `42`), groups them into 15 recordings of 10 bins each, and writes CSV files under:

```text
George_Selected_15_Recordings_Anes_Day2/
  <region>/<state>/Recording_01/bin_01.csv
  ...
  <region>/<state>/Recording_15/bin_10.csv
```

Each exported recording represents 20 seconds of ECoG data, assembled from ten randomly selected 2-second bins. The script stops with an error if a region/state has fewer than 150 bins, ensuring that every exported group has the same size.

## Python dependencies

```bash
pip install numpy scipy pandas matplotlib
```

`matplotlib` is imported by the preprocessing module but is not used in the current workflow.

## Project files

- `pre_process.py` — raw-data loading, resampling, rereferencing, notch filtering, state selection, and binning.
- `main_process.py` — ROI definitions and processing of the configured sessions.
- `make_recordings.py` — random bin selection and CSV export of the 15 recordings per region/state.