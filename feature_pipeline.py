"""
An example feature extraction pipeline.

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
__date__ = "July 2021"


import os
import sys

import lpne


USAGE = "Usage:\n$ python feature_pipleline.py <experiment_directory>"
LFP_SUBDIR = 'Data'
CHANS_SUBDIR = 'CHANS'
FEATURE_SUBDIR = 'features'
LFP_SUFFIX = '_LFP.mat'
CHANS_SUFFIX = '_CHANS.mat'
FS = 1000



if __name__ == '__main__':
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

    # Get the filenames.
    lfp_fns, chans_fns = lpne.get_lfp_chans_filenames(lfp_dir, chans_dir)

    for file_num in range(len(lfp_fns)):
        # Load LFP data.
        lfps = lpne.load_lfps(lfp_fns[file_num])

        # Plot the LFPs for fun.
        if file_num == 0:
            lpne.plot_lfps(
                    lfps,
                    t1=1.0,
                    t2=5.0,
                    fs=FS,
                    window_duration=2.0,
                    fn='example_lfps.pdf',
            )

        # Get the default channel grouping.
        channel_map = lpne.get_default_channel_map(list(lfps.keys()))

        # Load the contents of a file to determine which channels to remove.
        to_remove = lpne.get_removed_channels_from_file(chans_fns[file_num])

        # Remove these channels.
        channel_map = lpne.remove_channels(channel_map, to_remove)

        # Average channels in the same region together.
        lfps = lpne.average_channels(lfps, channel_map)

        # # ???
        # lfps = lpne.normalize_lfps(lfps)

        # Make features.
        features = lpne.make_features(lfps)

        # Plot the first window for fun.
        lpne.plot_power(features['power'][0], features['rois'])

        # Save features.
        fn = os.path.split(lfp_fns[file_num])[-1][:-len(LFP_SUFFIX)] + '.npy'
        fn = os.path.join(feature_dir, fn)
        lpne.save_features(features, fn)



###
