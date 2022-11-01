"""
Trying to make a more automated experimental pipeline.

A work in progress!

TODO: put the functions here into the main package
TODO: add the body of the __main__ block to the main package
TODO: default experiment parameters should be shipped with the code
TODO: check for duplicate mice on different days
TODO: group the parameters differently or add kwargs to functions?
TODO: remove channels that aren't in the channel map
"""
__date__ = "October - November 2022"


import numpy as np
import os
import sys
import warnings
import yaml

import lpne


FEATURE_SUFFIX = ".npy"
USAGE = "Usage:\n$ python script.py <experiment_directory>"


def get_all_fns(
    exp_dir,
    chans_subdir=None,
    feature_subdir=None,
    label_subdir=None,
    lfp_subdir=None,
    chans_suffix=None,
    label_suffix=None,
    lfp_suffix=None,
    strict_checking=True,
    **params,
):
    """
    Return the corresponding CHANS, feature, label, and LFP filenames.

    TODO: implement single CHANS file for multiple mouse files case

    Raises
    ------
    * AssertionError if ...
    * UserWarning if ...

    Parameters
    ----------
    exp_dir : str
        Experiment directory
    chans_subdir : str
        CHANS subdirectory
    feature_subdir : str
        Feature subdirectory
    label_subdir : str
        Label subdirectory
    lfp_subdirectory : str
        LFP subdirectory
    chans_suffix : str
        Common suffix for CHANS files
    label_suffix : str
        Common suffix for label files
    lfp_suffix : str
        Common suffix for LFP files
    strict_checking : bool, optional
        Toggles whether to throw an error or a warning if there are mismatches between
        the CHANS, label, and LFP directories. The feature directory is not checked to
        facilitate the case where the features haven't been calculated yet.

    Returns
    -------
    chans_fns : list of str
        CHANS filenames
    feature_fns : list of str
        Feature filenames
    label_fns : list of str
        Label filenames
    lfp_fns : list of str
        LFP filenames
    """
    # Make sure all the directories exist. Make the feature directory if it doesn't.
    assert chans_subdir is not None, "chans_subdir must be specified!"
    assert feature_subdir is not None, "feature_subdir must be specified!"
    assert label_subdir is not None, "label_subdir must be specified!"
    assert lfp_subdir is not None, "lfp_subdir must be specified!"
    assert os.path.exists(exp_dir), f"Experiment directory '{exp_dir}' doesn't exist!"
    chans_dir = os.path.join(exp_dir, chans_subdir)
    assert os.path.exists(chans_dir), f"CHANS directory '{chans_dir}' doesn't exist!"
    feature_dir = os.path.join(exp_dir, feature_subdir)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    label_dir = os.path.join(exp_dir, label_subdir)
    assert os.path.exists(label_dir), f"Label directory '{label_dir}' doesn't exist!"
    lfp_dir = os.path.join(exp_dir, lfp_subdir)
    assert os.path.exists(label_dir), f"LFP directory '{lfp_dir}' doesn't exist!"
    assert isinstance(chans_suffix, str) and len(chans_suffix) > 0
    assert isinstance(label_suffix, str) and len(label_suffix) > 0
    assert isinstance(lfp_suffix, str) and len(lfp_suffix) > 0

    # Figure out the missing/extra files.
    j = len(chans_suffix)
    chans_fns = [i[:-j] for i in os.listdir(chans_dir) if i.endswith(chans_suffix)]
    chans_fns = np.array(chans_fns)
    j = len(label_suffix)
    label_fns = [i[:-j] for i in os.listdir(label_dir) if i.endswith(label_suffix)]
    label_fns = np.array(label_fns)
    j = len(lfp_suffix)
    lfp_fns = [i[:-j] for i in os.listdir(lfp_dir) if i.endswith(lfp_suffix)]
    lfp_fns = np.array(lfp_fns)
    kw = dict(invert=True, assume_unique=True)
    chan_label_f = chans_fns[np.isin(chans_fns, label_fns, **kw)]
    label_chan_f = label_fns[np.isin(label_fns, chans_fns, **kw)]
    chan_lfp_f = chans_fns[np.isin(chans_fns, lfp_fns, **kw)]
    lfp_chan_f = lfp_fns[np.isin(lfp_fns, chans_fns, **kw)]
    label_lfp_f = label_fns[np.isin(label_fns, lfp_fns, **kw)]
    lfp_label_f = lfp_fns[np.isin(lfp_fns, label_fns, **kw)]
    temp = [
        chan_label_f,
        label_chan_f,
        chan_lfp_f,
        lfp_chan_f,
        label_lfp_f,
        lfp_label_f,
    ]
    lens = [len(i) for i in temp]
    msg = ""
    if lens[0] > 0:
        msg += f"Found {lens[0]} files in CHANS not in labels! "
    if lens[1] > 0:
        msg += f"Found {lens[1]} files in labels not in CHANS! "
    if lens[2] > 0:
        msg += f"Found {lens[2]} files in CHANS not in LFPs! "
    if lens[3] > 0:
        msg += f"Found {lens[3]} files in LFPs not in CHANS! "
    if lens[4] > 0:
        msg += f"Found {lens[4]} files in labels not in LFPs! "
    if lens[5] > 0:
        msg += f"Found {lens[5]} files in LFPs not in labels! "
    msg = msg[:-1] if msg != "" else msg  # cut off the last space character
    if strict_checking:
        assert sum(lens) == 0, msg
    elif len(msg) > 0:
        warnings.warn(msg)

    # Figure out the common group of files.
    fns = np.intersect1d(chans_fns, label_fns)
    fns = np.intersect1d(fns, lfp_fns)  # guaranteed to be sorted
    assert len(fns) > 0, (
        f"Found no filenames in common between CHANS, label, and " f"LFP directories!"
    )
    chans_fns = [os.path.join(chans_dir, i + chans_suffix) for i in fns]
    feature_fns = [os.path.join(feature_dir, i + FEATURE_SUFFIX) for i in fns]
    label_fns = [os.path.join(label_dir, i + label_suffix) for i in fns]
    lfp_fns = [os.path.join(lfp_dir, i + lfp_suffix) for i in fns]
    return chans_fns, feature_fns, label_fns, lfp_fns


def infer_groups_from_fns(fns):
    """
    Infer groups from the filenames.

    The expected filename format is: */Mouse<id>_<date>_*.mat

    Unique combinations of <id> and <date> are assigned to different groups. Both <id>
    and <date> are assumed to contain no underscores.

    Raises
    ------
    * AssertionError if any of the filename are improperly formatted.

    Parameters
    ----------
    fns : list of str
        Filenames

    Returns
    -------
    groups : list of int
        Inferred group for each filename. The groups are zero-indexed and sorted by the
        tuple ``(id, date)``.
    group_map : dict
        Maps group name to integer group
    """
    # First check to make sure all the filenames are in the correct format.
    fns = [os.path.split(fn)[1] for fn in fns]
    group_tuples = []
    for fn in fns:
        assert fn.startswith("Mouse"), f"Filename {fn} doesn't start with 'Mouse'!"
        temp = fn.split("_")
        assert len(temp) >= 3, f"Expected filename {fn} to have at least 2 underscores!"
        group_tuples.append((temp[0][len("Mouse") :] + "_" + temp[1]))
    # Sort the groups and make map.
    unique_group_tuples = np.unique(group_tuples).tolist()  # guaranteed sorted
    groups = [unique_group_tuples.index(tup) for tup in group_tuples]
    group_map = dict(zip(group_tuples, groups))
    return groups, group_map


def load_channel_map(fn):
    """
    Load the channel map from the file.

    Parameters
    ----------
    fn : str
        Filename of the channel map file

    Returns
    -------
    channel_map : dict
        Maps channel names (str) to region names (str)
    """
    if fn.endswith(".csv"):
        channel_map = {}
        with open(fn, "r") as f:
            for ln, line in enumerate(f):
                parts = [p.strip() for p in line.split(",") if len(p.strip()) > 0]
                if len(parts) == 0:
                    continue
                assert len(parts) == 2, f"Expected two columns on line {ln+1}!"
                assert parts[0] not in channel_map, f"Duplicate entry on line {ln+1}!"
                channel_map[parts[0]] = parts[1]
        return channel_map
    else:
        raise NotImplementedError(f"Cannot load the channel map file: {fn}")


if __name__ == "__main__":
    # Check the input argument.
    if len(sys.argv) != 2:
        quit(USAGE)
    exp_dir = sys.argv[1]

    # Load the parameters.
    try:
        stream = open("params.yaml", "r")
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
    except:
        quit("Cannot load params.yaml!")
    print("Parameters:\n\n" + yaml.dump(params))

    # Load feature, CHANS, and label filenames with all the checks.
    chans_fns, feature_fns, label_fns, lfp_fns = get_all_fns(exp_dir, **params["file"])
    groups, group_map = infer_groups_from_fns(lfp_fns)

    if params["pipeline"]["make_features"]:
        # Load the channel map.
        channel_map_fn = os.path.join(exp_dir, params["file"]["channel_map_fn"])
        channel_map = load_channel_map(channel_map_fn)
        # Make the features for each filename.
        for file_num in range(len(lfp_fns)):
            print(f"File: {file_num+1}/{len(lfp_fns)}:", lfp_fns[file_num])
            # Load the LFPs.
            lfps = lpne.load_lfps(lfp_fns[file_num])
            # Remove the bad channels marked in the CHANS file.
            lfps = lpne.remove_channels_from_lfps(lfps, chans_fns[file_num])
            # Filter the LFPs.
            lfps = lpne.filter_lfps(
                lfps,
                params["preprocess"]["fs"],
                lowcut=params["preprocess"]["filter_lowcut"],
                highcut=params["preprocess"]["filter_highcut"],
            )
            if params["preprocess"]["remove_outliers"]:
                # Mark outliers with NaNs.
                lfps = lpne.mark_outliers(
                    lfps,
                    params["preprocess"]["fs"],
                    lowcut=params["preprocess"]["outlier_lowcut"],
                    highcut=params["preprocess"]["filter_highcut"],
                    mad_threshold=params["preprocess"]["outlier_mad_threshold"],
                )
                # Print outlier summary.
                msg = lpne.get_outlier_summary(
                    lfps,
                    params["preprocess"]["fs"],
                    params["preprocess"]["window_duration"],
                )
                print(msg)
            # Average channels and combine outliers in the same group.
            lfps = lpne.average_channels(lfps, channel_map)
            # Make features.
            features = lpne.make_features(
                lfps,
                fs=params["preprocess"]["fs"],
                min_freq=params["preprocess"]["feature_min_freq"],
                max_freq=params["preprocess"]["feature_max_freq"],
                window_duration=params["preprocess"]["window_duration"],
                window_step=params["preprocess"]["window_step"],
                max_n_windows=params["preprocess"]["max_n_windows"],
                directed_spectrum=params["preprocess"]["directed_spectrum"],
                csd_params=params["preprocess"]["csd_params"],
            )
            # Save the features.
            lpne.save_features(features, feature_fns[file_num])

    # Load all the features and labels.

    # TODO: clean this up by editing ``load_features_and_labels``
    def group_func(fn):
        for key in group_map:
            if key in os.path.split(fn)[1]:
                return group_map[key]
        raise NotImplementedError(fn)

    features, labels, rois, groups = lpne.load_features_and_labels(
        feature_fns,
        label_fns,
        group_func=group_func,
    )

    print("features", features.shape)
    print("rois", rois)
    quit()

    # Normalize the features.
    # TODO: add normalization mode
    features = lpne.normalize_features(features)  # [b,r(r+1)//2,f]
    features = lpne.unsqueeze_triangular_array(features, 1)  # [b,r,r,f]
    features = np.transpose(features, [0, 3, 1, 2])  # [b,f,r,r]

    # Make some summary plots showing the different classes and mice.
    pass

    # Do some cross-validation to estimate generalization and train a single model.

    # Save the model.

    # See how well we predict on the test set.

    # Summarize the reconstruction quality on the train and test sets.

    # Plot the predictive factors.


if __name__ == "__main__":
    pass


###
