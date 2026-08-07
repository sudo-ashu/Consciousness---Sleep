import os
from pre_process import main_function

# Region Of Interest
ROI = {
    "WB": list(range(1,129)),
    "MP": [52, 57, 60, 63],
    "LP": [14, 25, 37, 46, 48],
    "PM": [5, 6, 39, 50],
    "MS": [18, 41, 42, 30],
    "PC": [1, 2, 10, 11, 21, 32],
    "TC": [67, 68, 80, 92, 93, 94],
    "HV": [13, 72, 73, 84, 85, 98],
    "LC": [88, 89, 101, 102, 112, 113]
}

session_folders = ["KT_George_Day_2/Session1", "KT_George_Day_2/Session2", "KT_George_Day_2/Session3"]

region_state_bins = {}

for region in ROI:
    print(f"\n{'='*25}")
    print(f"Processing Region: {region}")
    print(f"{'='*25}")

    region_state_bins[region] = main_function(session_folders, ROI, region)

print(region_state_bins.keys())