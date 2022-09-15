"""
Channel maps are used to determine which channels to average together.

"""
__date__ = "July 2021 - May 2022"


import numpy as np
import pandas as pd
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

    Channels (keys) in ``lfps`` that map to the same group name (value) in
    ``channel_map`` will be averaged together and named by the group name.

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
            # Find NaNs and replace them with zeros to calculate an average.
            nan_masks = [np.isnan(trace) for trace in avg]
            nan_mask = np.minimum(sum(nan_masks), 1)
            for i in range(len(avg)):
                avg[i][nan_masks[i]] = 0.0
            avg = sum(avg) / len(avg)
            # Reintroduce the NaNs into the averaged LFP.
            avg[nan_mask > 0] = np.nan
            out_lfps[grouped_roi] = avg
    return out_lfps

def get_excel_channel_map(channels,excel_path):
    """
    Load predifined channel map from an excel file.

    Raises
    ----------
    *   UserWarning if a channel present in the LFP is not present in
        the excel channel map.
    Parameters
    ----------
    excel_path : str
        Path to excel file

    Returns
    ---------
    channel_map : dict
        Maps individual channel names to grouped channel names.
    """
    f = pd.read_excel(excel_path,index_col=False,header=None)
    channel_map_full = f.set_index(0).to_dict()[1]

    channel_map = {}

    #Check that all channels are in the channel map
    for channel in channels:
        if channel not in list(channel_map_full.keys()):
            warnings.warn(
                f"{channel} exists in LFP files but is not present in"
                f"channel map file {excel_path}"
            )
        else:
            channel_map[channel] = channel_map_full[channel]

    return channel_map


def get_default_channel_map(channels, combine_hemispheres=True):
    """
    Make a default channel map.

    Raises
    ------
    * UserWarning if a channel doesn't have an underscore or a hemisphere
      indication in its name. The channel is ignored in this case.

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
            warnings.warn(
                f"Expected an underscore in channel name: {channel}"
                f", ignoring."
            )
            continue
        split_str = channel.split('_')
        try:
            temp = int(split_str[-1])
        except ValueError:
            warnings.warn(
                f"Unexpected channel name: {channel}"
                f", channel should end in a number. Ignoring."
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


def remove_channels_from_lfps(lfps, fn):
    """
    Remove channels specified in the CHANS file from the LFPs.
    
    Parameters
    ----------
    lfps : dict
        Maps ROI names to LFP waveforms.
    fn : str
        CHANS file filename

    Returns
    -------
    lfps : dict
        Maps ROI names to LFP waveforms
    """
    to_remove = get_removed_channels_from_file(fn)
    for roi in to_remove:
        if roi in lfps:
            del lfps[roi]
    return lfps


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
            warnings.warn(f"Channel {channel} is not present in LFPS!")



if __name__ == '__main__':
    pass



###
