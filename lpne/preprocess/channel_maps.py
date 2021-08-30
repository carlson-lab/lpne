"""
Channel maps are used to determine which channels to average together

"""
__date__ = "July - August 2021"


import numpy as np
from scipy.io import loadmat
import warnings


IGNORED_KEYS = [
    '__header__',
    '__version__',
    '__globals__',
]
"""Ignored keys in the LFP data file"""



def average_channels(lfps, channel_map, check_channel_map=True):
    """
    Average different channels in the the same region.

    Expected behavior -- ... finish this

    Parameters
    ----------
    lfps : dict
        Maps ROI names to LFP waveforms.
    channel_map : dict
        Maps ROI names to grouped ROI names.
    check_channel_map : bool, optional
        Checks whether all the channels in the channel map are present in the
        LFPs.

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
                avg.append(lfps[roi].flatten())
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


def get_default_channel_map(channels, combine_hemispheres=True):
    """
    Make a default channel map...

    Raises
    ------
    * UserWarning if ...

    Parameters
    ----------
    channels : list of str
        Names of channels.
    combine_hemispheres : bool, optional
        Combine channels from the left and right hemispheres.

    Returns
    -------
    channel_map : dict
        Maps individual channel names to grouped channel names.
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
            if combine_hemispheres:
                if len(split_str) < 3 or split_str[-2] not in ['L', 'R']:
                    warnings.warn(
                        f"Unexpected channel name: {channel}"
                        f", no hemisphere indication. Ignoring."
                    )
                    continue
                else:
                    grouped_roi = '_'.join(split_str[:-2])
            else:
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
    channel_map : dict
        The input channel map with deleted keys.
    """
    assert isinstance(channel_map, dict)
    assert isinstance(to_remove, list)
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
        # except: for old .mat files in hdf5 format...
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



if __name__ == '__main__':
    pass



###
