"""
Data utilities.

"""
__date__ = "July 2021"

import numpy as np
from scipy.io import loadmat
import warnings


IGNORED_KEYS = [
    '__header__',
    '__version__',
    '__globals__',
]
"""Ignored keys in the LFP data file."""



def load_data(fn):
    """

    Parameters
    ----------
    fn : str
        File containing LFP data.

    Returns
    -------
    lfps : dict
        Maps ROI names to LFP waveforms.
    """
    assert isinstance(fn, str)
    if fn.endswith('.mat'):
        # try:
        data = loadmat(fn)
        # except: for old .mat files...
    else:
        raise NotImplementedError(f"Cannot load file: {fn}")
    return data



def average_channels(lfps, channel_map, check_channel_map=True):
    """
    Average different channels in the the same region.

    Expected behavior -- check everything in channel map is used

    Parameters
    ----------
    lfps : dict
        Maps ROI names to LFP waveforms.
    channel_map : dict
        Maps ROI names to grouped ROI names.
    check_channel_map : bool, optional
        Checks for expected behavior...

    Returns
    -------
    lfps : dict
        Maps ROI names to LFP waveforms.
    """
    if check_channel_map:
        _check_channel_map(lfps, channel_map)
    out_lfps = {}
    for grouped_roi in np.unique(list(channel_map.values())):
        avg = []
        for roi in lfps.keys():
            if roi in channel_map and channel_map[roi] == grouped_roi:
                avg.append(lfps[roi])
        if len(avg) == 0:
            warnings.warn(f"No channels to make grouped channel {grouped_roi}!")
        else:
            avg = sum(avg) / len(avg)
            out_lfps[grouped_roi] = avg
    return out_lfps



def _check_channel_map(lfps, channel_map):
    """
    Check that every channel in `channel_map` is in `lfps`.

    Warnings
    --------
    * Whenever there is a channel in `channel_map` that isn't in `lfps`.

    Parameters
    ----------
    lfps : dict
        Maps ROI names to LFP waveforms.
    channel_map : dict
        Maps ROI names to grouped ROI names.
    """
    for channel in channel_map:
        if channel not in lfps:
            warnings.warn(f"Channel {channel} is not present!")


def get_default_channel_map(channels):
    """
    Make a default channel map...

    Raises
    ------
    * UserWarning if ...

    Parameters
    ----------

    Returns
    -------
    channel_map : dict
    """
    channel_map = {}
    for channel in channels:
        if channel in IGNORED_KEYS:
            continue
        if '_' not in channel:
            warnings.warn(f"Expected an underscore in channel name: {channel}")
        else:
            split_str = channel.split('_')
            try:
                temp = int(split_str[-1])
            except ValueError:
                warnings.warn(
                    f"Unexpected channel name: {channel}"
                    f", ignoring."
                )
                continue
            grouped_roi = '_'.join(split_str[:-1])
            channel_map[channel] = grouped_roi
    return channel_map


def remove_channels(channel_map, to_remove):
    """
    Remove channels from the channel map.

    Raises
    ------
    * UserWarning if a channel to remove isn't in the channel map.

    Parameters
    ----------
    channel_map : dict
        ...
    to_remove : list
        List of channels to remove.

    Returns
    -------
    """
    for channel in to_remove:
        if channel not in channel_map:
            warnings.warn(f"Channel {channel} isn't in the channel map!")
            continue
        del channel_map[channel]
    return channel_map


def get_removed_channels_from_file(fn):
    """
    Load a list of removed channels from a file.

    Raises
    ------
    * NotImplementedError if the file format isn't supported.

    Parameters
    ----------
    fn : str
        Filename

    Returns
    -------
    to_remove : list of str
        List of channels to remove.
    """
    assert isinstance(fn, str)
    if fn.endswith('.mat'):
        # try:
        data = loadmat(fn)
        # except: for old .mat files...
        assert('CHANNAMES' in data), f"{fn} must contain CHANNAMES!"
        assert('CHANACTIVE' in data), f"{fn} must contain CHANACTIVE!"
        channel_active = data['CHANACTIVE'].flatten()
        channel_names = np.array(
                [str(i[0]) for i in data['CHANNAMES'].flatten()],
        )
        idx = np.argwhere(channel_active == 0).flatten()
        return channel_names[idx].tolist()
    else:
        raise NotImplementedError(f"Cannot load file: {fn}")



def save_features(features, fn):
    """
    Save the features to the given filename.

    Raises
    ------
    * NotImplementedError if `fn` is an unsupported file type.

    Paramters
    ---------
    features : dict
        ...
    fn : str
        Where to save the data.
    """
    assert isinstance(fn, str)
    if fn.endswith('.npy'):
        np.save(fn, features)
    else:
        raise NotImplementedError(f"Unsupported file type: {fn}")



if __name__ == '__main__':
    lfp_fn = 'test_data/example_LFP.mat'
    chans_fn = 'test_data/example_CHANS.mat'
    channel_map = {
            'D_Hipp_01': 'D_Hipp',
            'D_Hipp_02': 'D_Hipp',
            'Foo': 'Bar',
    }

    data = load_data(lfp_fn)
    print(list(data.keys()))
    print('Amy_CeA_L_01' in data)
    print('Cx_OF_L_01' in data)
    print('Thal_MD_L_02' in data)

    channel_map = get_default_channel_map(list(data.keys()))

    to_remove = get_removed_channels_from_file(chans_fn)

    channel_map = remove_channels(channel_map, to_remove)

    data = average_channels(data, channel_map)
    print(list(data.keys()))




###
