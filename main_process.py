import os
from pre_process import load_session, preprocess, experiment_info, make_bins

fs = 200

# ROI - Region of Interest
ROI = {
    "Frontal": [1,2,3,4,8,9,10,11,12,18,19,20,21,22,29,30,31,32,33,40,41,42,43,44],
    "Parietal": [45,46,47,48,49,50,51,56,57,58,59,60,63,64],
    "Temporal": list(range(65,109)) + [52,53,54,55,61,62],
    "Occipital": list(range(109,129)),
    "Posterior_Hot_Zone": [ 56,57,58,59,60,63,64,] + list(range(101,129))
}


all_region_bins = {}
# works on the meta data
session_folders = ["KT_George_Day_1/Session1", "KT_George_Day_1/Session2", "KT_George_Day_1/Session3"]

for region, electrodes in ROI.items():
    channel_idx = [e-1 for e in electrodes]
    state_bins = {}

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
                start_idx = int(start_time * fs)
                end_idx = int(end_time * fs)

                segment = data_filt[:, start_idx:end_idx]
                bins = make_bins(segment)

                state_bins[state].extend(bins)

    all_region_bins[region] = state_bins

print(all_region_bins.keys())