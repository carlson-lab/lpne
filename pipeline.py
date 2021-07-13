"""
An example feature extraction pipeline.

Expected directory structure:

exp_dir
.
.
.


"""
__date__ = "July 2021"


import os
import sys

import lpne


USAGE = "Usage:\n$ python pipeline.py <experiment_directory>"
LFP_SUBDIR = 'Data'
CHANS_SUBDIR = 'CHANS'
FEATURE_SUBDIR = 'features'
LFP_SUFFIX = '_LFP.mat'



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

    # Get the filenames straight.
    lfp_fns = [fn for fn in sorted(os.listdir(lfp_dir))]
    lfp_fns = [
        os.path.join(lfp_dir,fn) for fn in lfp_fns if fn.endswith('.mat')
    ]
    chans_fns = [fn for fn in sorted(os.listdir(chans_dir))]
    chans_fns = [
        os.path.join(chans_dir,fn) for fn in chans_fns if fn.endswith('.mat')
    ]
    assert len(lfp_fns) == len(chans_fns)
    for i in range(len(lfp_fns)):
        assert lfp_fns[i].endswith(LFP_SUFFIX)
        assert os.path.split(lfp_fns[i][-1]) == os.path.split(chans_fns[i][-1])

    for file_num in range(len(lfp_fns)):
        # Load LFP data.
        lfps = lpne.load_data(lfp_fns[file_num])

        # Get the default channel grouping.
        channel_map = lpne.get_default_channel_map(list(lfps.keys()))

        # Load the contents of a file to determine which channels to remove.
        to_remove = lpne.get_removed_channels_from_file(chans_fns[file_num])

        # Remove these channels.
        channel_map = lpne.remove_channels(channel_map, to_remove)

        # Average channels in the same region together.
        lfps = lpne.average_channels(lfps, channel_map)

        # Make features.
        features = lpne.make_features(lfps)

        # Save features.
        fn = os.path.split(lfp_fns[file_num])[-1][:len(LFP_SUFFIX)] + '.npy'
        fn = os.path.join(feature_dir, fn)
        lpne.save_features(features, fn)

###
