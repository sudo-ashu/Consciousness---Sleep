import os
import pandas as pd
import random
from main_process import region_state_bins

random.seed(42)
selected_bins = {}

for region in region_state_bins:
    selected_bins[region] = {}

    for state, bins in region_state_bins[region].items():
        if len(bins) < 150:
            raise ValueError(f"{region} - {state} has only {len(bins)} bins (<150)")
        selected_bins[region][state] = random.sample(bins, 150)


recordings = {}
for region in selected_bins:
    recordings[region] = {}

    for state in selected_bins[region]:
        bins = selected_bins[region][state]
        recordings[region][state] = [bins[i:i+10] for i in range(0, 150, 10)]


# Now making 15 recordings per state
base_folder = "Chibi_SLP_Rec_Day2"
os.makedirs(base_folder, exist_ok=True)

for region in recordings:
    region_folder = os.path.join(base_folder, region)
    os.makedirs(region_folder, exist_ok=True)

    for state in recordings[region]:
        state_folder = os.path.join(region_folder, state)
        os.makedirs(state_folder, exist_ok=True)

        for rec_idx, recording in enumerate(recordings[region][state], start=1):
            rec_folder = os.path.join(state_folder, f"Recording_{rec_idx:02d}")
            os.makedirs(rec_folder, exist_ok=True)

            for bin_idx, bin_data in enumerate(recording, start=1):
                filename = os.path.join(rec_folder,f"bin_{bin_idx:02d}.csv")
                pd.DataFrame(bin_data).to_csv(filename, index=False)