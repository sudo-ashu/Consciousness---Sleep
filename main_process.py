import os
from pre_process import main_function

# Region Of Interest
ROI = {
    "WB": list(range(1,129)),
    "MP": [52],
    "LP": [25],
    "PM": [39],
    "MS": [30],
    "PC": [11],
    "TC": [94],
    "HV": [85],
    "LC": [102]
}

session_folders = ["KT_George_Day_1/Session1", "KT_George_Day_1/Session2"]

region_state_bins = {}

for region in ROI:
    print(f"\n{'='*25}")
    print(f"Processing Region: {region}")
    print(f"{'='*25}")

    region_state_bins[region] = main_function(session_folders, ROI, region)

print(region_state_bins.keys())