# %%
import os
import scipy.io
import h5py
import numpy as np
import mne
import glob
import re
import matplotlib.pyplot as plt

# %%
# --- Configuration ---
data_folder = 'monkey_anesthesia_sleep_data\Session1'
time_file_name = 'ECoGTime.mat'
condition_file_name = 'Condition.mat'

# %%
# --- Helper Function for Natural Sorting ---
def numerical_sort_key(file_path):
    # Extracts numbers from filename so "Ch2" comes before "Ch10"
    numbers = re.findall(r'\d+', os.path.basename(file_path))
    return int(numbers[0]) if numbers else 0

# %%
# --- 1. Load the Time Vector First ---
time_path = os.path.join(data_folder, time_file_name)
try:
    # Try standard load
    time_mat = scipy.io.loadmat(time_path)
    # Check common keys for time
    t_key = next(k for k in time_mat.keys() if k not in ['__header__', '__version__', '__globals__'])
    time_vec = time_mat[t_key].flatten()
except:
    # Fallback to h5py if needed (same as before)
    import h5py
    with h5py.File(time_path, 'r') as f:
        t_key = list(f.keys())[0]
        time_vec = np.array(f[t_key]).flatten()

print(time_vec.shape)


# %%
# Calculate Sampling Rate
sfreq = 1 / (time_vec[1] - time_vec[0])
print(f"Sampling Frequency: {sfreq:.2f} Hz")

# --- 2. Load and Stack Channel Files ---
# Find all .mat files except the Time file
all_files = glob.glob(os.path.join(data_folder, '*.mat'))
channel_files = [f for f in all_files if time_file_name not in f and 'Map' and 'Condition' not in f]

# SORT them numerically
channel_files.sort(key=numerical_sort_key)

print(f"Found {len(channel_files)} channel files.")
print(f"First 3 files: {[os.path.basename(f) for f in channel_files[1:4]]}")


# %%
# Load loop
data_list = []
for fpath in channel_files:
    try:
        mat = scipy.io.loadmat(fpath)
        # Find the data key (ignoring headers)
        key = next(k for k in mat.keys() if k not in ['__header__', '__version__', '__globals__'])
        data = mat[key].flatten() # Ensure it's a 1D array
        data_list.append(data)
    except Exception as e:
        print(f"Error loading {fpath}: {e}")

# Stack into (128, n_samples) matrix
# This matrix is the foundation for Network Analysis later
raw_array = np.vstack(data_list)
print(f"Final Data Shape: {raw_array.shape}")

# %%
# --- 3. Create MNE Object ---
n_channels = len(data_list)
ch_names = [f'Ch{i+1}' for i in range(n_channels)]
ch_types = ['ecog'] * n_channels

# Initialize Info
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Create Raw object
raw = mne.io.RawArray(raw_array, info)

# --- 4. Plotting ---
# A. Butterfly Plot (Time Series)
# This overlays all channels. Look for flat lines (dead channels) or huge spikes.
raw.plot(duration=5, n_channels=n_channels, scalings='auto', title="128-Channel ECoG")

# B. Plot Power Spectral Density (PSD)
# Useful to see if you have clean Alpha/Beta peaks or just noise
raw.compute_psd(fmax=100).plot()

# %%
file_path = 'monkey_anesthesia_sleep_data\Session1\Condition.mat'

if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
else:
    print(f"Loading {file_path}...")
    cond_data = scipy.io.loadmat(file_path)

    # Debug: Confirm it is a dictionary now
    print(f"Type of cond_data: {type(cond_data)}") # Should say <class 'dict'>

    # 4. Extract the keys automatically
    # We ignore internal keys like '__header__', '__version__'
    keys = [k for k in cond_data.keys() if not k.startswith('__')]
    print(f"Valid keys found: {keys}")

    if len(keys) > 0:
        # 5. Extract the labels using the first valid key
        target_key = keys[0]
        cond_labels = cond_data[target_key].flatten()
        
        print(f"Success! Extracted '{target_key}' with shape {cond_labels.shape}")
        print(f"Unique values: {np.unique(cond_labels)}")
    else:
        print("Error: No valid variables found in this .mat file.")

# %%
idx_open_start = cond_labels[0]
idx_open_end = cond_labels[1]
idx_close_start = cond_labels[2]
idx_close_end = cond_labels[3]

eyes_open = raw_array[:, idx_open_start : idx_open_end]
eyes_closed = raw_array[:, idx_close_start : idx_close_end]

print(f"Eyes Open Duration:   {eyes_open.shape[1] / 1000 :.2f} seconds")
print(f"Eyes Closed Duration: {eyes_closed.shape[1] / 1000 :.2f} seconds")

# %%
# Potting the difference
sfreq = 1000.0 

# Create Figure
fig, ax = plt.subplots(figsize=(10, 6))

# Helper to plot
def plot_psd(data, label, color):
    # Create temp MNE object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='ecog')
    raw_temp = mne.io.RawArray(data, info, verbose=False)
    
    # Compute PSD (1-60Hz)
    spectrum = raw_temp.compute_psd(fmin=1, fmax=60, n_fft=2048)
    psd, freqs = spectrum.get_data(return_freqs=True)
    
    # Average across all 128 channels
    mean_psd = np.mean(psd, axis=0)
    
    # Plot in dB
    ax.plot(freqs, 10 * np.log10(mean_psd), color=color, label=label, linewidth=2)

# Plot both conditions
plot_psd(eyes_open,   'Awake: Eyes Open',   'blue')
plot_psd(eyes_closed, 'Awake: Eyes Closed', 'red')

# Styling
ax.set_title('Alpha Block Check: Eyes Open vs Closed')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (dB)')
ax.legend()
ax.axvline(10, color='gray', linestyle='--', alpha=0.5, label='10Hz Alpha') # Mark 10Hz
ax.grid(True)

plt.show()


