"""
Usage: python feature_pipeline_tst.py <path_to_directory>

The first part of the example implementation of the feature and prediction pipeline on the 
open source TST dataset. This part makes the features, the second part does the prediction. 

First, we will make features and then we will do predictions on those features. 
The downloaded directory already has features, so the output will be the same as what is already there.

Required directory structure:
::
    exp_dir
    │
    ├── Data
    │   ├── foo_LFP.mat
    │   ├── bar_LFP.mat
    │   └── baz_LFP.mat
    │
    ├── CHANS
    │   ├── foo_CHANS.mat
    │   ├── bar_CHANS.mat
    │   └── baz_CHANS.mat
    │
    └── features
        ├── foo.npy
        ├── bar.npy
        └── baz.npy

The subdirectories `Data` and `CHANS` are inputs to this pipeline and the
subdirectory `features` is the output.
"""


__date__ = "September 2022 - October 2022"

import os
import sys

import lpne

USAGE = "Usage:\n$ python feature_pipeline_tst.py <experiment_directory>"
LFP_SUBDIR = "Data"
CHANS_SUBDIR = "CHANS"
FEATURE_SUBDIR = "features"
LFP_SUFFIX = "_LFP.mat"
CHANS_SUFFIX = "_CHANS.mat"
FS = 1000  # Samplerate (Hz)
WINDOW_DURATION = 1  # Window duration (s)
DIR_SPEC = True  # Whether to calculate directed spectrum features
LOWCUT = 0.5  # Lowcut for bandpass filter (Hz)
OUTLIER_LOWCUT = 30.0  # Lowcut filter for outlier detection (Hz)
HIGHCUT = 200  # Highcut for bandpass filter (Hz)
MAX_FREQ = 55.0  # Maximum frequency for popwer features (Hz)
REMOVE_OUTLIERS = True  # Remove windows with outlier samples


if __name__ == "__main__":
    # Check input arguments.
    if len(sys.argv) != 2:
        quit(USAGE)
    exp_dir = sys.argv[1]
    assert os.path.exists(exp_dir)
    lfp_dir = os.path.join(exp_dir, LFP_SUBDIR)
    assert os.path.exists(lfp_dir)
    chans_dir = os.path.join(exp_dir, CHANS_SUBDIR)
    assert os.path.exists(chans_dir)
    feature_dir = os.path.join(exp_dir, FEATURE_SUBDIR)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    # Get the LFP and CHANS filenames.
    lfp_fns, chans_fns = lpne.get_lfp_chans_filenames(lfp_dir, chans_dir)

    # For each file ...
    for file_num in range(len(lfp_fns)):

        # Load the LFPs.
        lfps = lpne.load_lfps(lfp_fns[file_num])

        # Remove the bad channels marked in the CHANS file.
        lfps = lpne.remove_channels_from_lfps(lfps, chans_fns[file_num])

        # Filter the LFPs.
        lfps = lpne.filter_lfps(lfps, FS, lowcut=LOWCUT, highcut=HIGHCUT)

        # Mark outliers with NaNs.
        lfps = lpne.mark_outliers(
            lfps,
            FS,
            lowcut=OUTLIER_LOWCUT,
            highcut=HIGHCUT,
        )

        # Print outlier summary.
        print(lpne.get_outlier_summary(lfps, FS, WINDOW_DURATION))

        # Get the channel map from the excel sheet
        channel_map = lpne.get_excel_channel_map(
            list(lfps.keys()), "~/Desktop/TST-Open-Source/ChannelNames.xlsx"
        )

        # Average channels according to the channel map.
        lfps = lpne.average_channels(lfps, channel_map)

        # Make features.
        features = lpne.make_features(
            lfps,
            window_duration=WINDOW_DURATION,
            directed_spectrum=DIR_SPEC,
            max_freq=MAX_FREQ,
        )

        # Save the features.
        fn = os.path.split(lfp_fns[file_num])[-1][: -len(LFP_SUFFIX)] + ".npy"
        fn = os.path.join(feature_dir, fn)
        lpne.save_features(features, fn)
