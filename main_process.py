import os
from pre_process import main_function

# Region Of Interest
ROI = {
    # "WB": list(range(1,129)),
    "MP": [52, 58, 56, 63], #, 57, 60, 63
    "LP": [35, 24, 37, 26, 48], #, 14, 37, 46, 48
    # "PM": [39], #, 6, 39, 50
    # "MS": [30], #, 41, 42, 30
    "PC": [68, 71, 83], #1, 2, 10, 11, 21, 32
    "TC": [11, 12, 42, 31, 20], #67, 68, 80, 92, 93, 94
    # "HV": [85], #13, 72, 73, 84, 85, 98
    # "LC": [102] #8, 89, 101, 102, 112, 113
}

session_folders = ["SLP_Chibi_Day2/Session1", "SLP_Chibi_Day2/Session2"]

region_state_bins = {}

for region in ROI:
    print(f"\n{'='*25}")
    print(f"Processing Region: {region}")
    print(f"{'='*25}")

    region_state_bins[region] = main_function(session_folders, ROI, region)

print(region_state_bins.keys())