"""
Channel maps are used to determine which channels to average together.

"""
__date__ = "July 2021 - February 2024"
__all__ = [
    "average_channels",
    "get_default_channel_map",
    "get_magic_channel_map",
    "remove_channels",
    "remove_channels_from_lfps",
    "get_removed_channels_from_file",
]


import numpy as np
from scipy.io import loadmat
import warnings

from .. import MATLAB_IGNORED_KEYS


def average_channels(
    lfps,
    channel_map,
    assert_onto=False,
    check_lfp_channels_in_map=True,
    check_map_channels_in_lfps=False,
    strict_checking=False,
):
    """
    Average different channels in the the same region.

    Channels (keys) in ``lfps`` that map to the same group name (value) in
    ``channel_map`` will be averaged together and named by the group name.

    Parameters
    ----------
    lfps : dict
        Maps ROI names to LFP waveforms
    channel_map : dict
        Maps ROI names to grouped ROI names
    assert_onto : bool, optional
        Assert that the map from channels to ROIs is onto, there is a channel in
        ``lfps`` that maps to every value in ``channel_map``.
    check_lfp_channels_in_map : bool, optional
        Check whether all the LFP channels are in the channel map
    check_map_channels_in_lfps : bool, optional
        Check whether all the map channels are in the LFPs
    strict_checking : bool, optional
        Whether to throw an error (if ``strict_checking`` is ``True``) or a warning
        (otherwise) when checking the channels.

    Returns
    -------
    lfps : dict
        Maps ROI names to LFP waveforms
    """
    # Check the channels.
    if check_lfp_channels_in_map:
        _check_lfp_channels_in_map(lfps, channel_map, strict_checking=strict_checking)
    if check_map_channels_in_lfps:
        _check_map_channels_in_lfps(lfps, channel_map, strict_checking=strict_checking)
    # Form each region LFP by averaging channels.
    out_lfps = {}
    for grouped_roi in np.unique(list(channel_map.values())):
        avg = []
        for roi in lfps.keys():
            if roi in channel_map and channel_map[roi] == grouped_roi:
                avg.append(lfps[roi].flatten())
        if len(avg) == 0:
            msg = f"No channels to make grouped channel: {grouped_roi}!"
            if assert_onto:
                assert False, msg
            else:
                warnings.warn(msg)
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


def get_default_channel_map(channels, combine_hemispheres=True):
    """
    Make a default channel map.

    Raises
    ------
    * ``UserWarning`` if a channel doesn't have an underscore or a hemisphere
      indication in its name. The channel is ignored in this case.

    Parameters
    ----------
    channels : list of str
        Names of channels
    combine_hemispheres : bool, optional
        Combine channels from the left and right hemispheres

    Returns
    -------
    channel_map : dict
        Maps individual channel names to grouped channel names
    """
    channel_map = {}
    for channel in channels:
        if channel in MATLAB_IGNORED_KEYS:
            continue
        if "_" not in channel:
            warnings.warn(
                f"Expected an underscore in channel name: {channel}" f", ignoring."
            )
            continue
        split_str = channel.split("_")
        try:
            temp = int(split_str[-1])
        except ValueError:
            warnings.warn(
                f"Unexpected channel name: {channel}"
                f", channel should end in a number. Ignoring."
            )
            continue
        if combine_hemispheres:
            if len(split_str) < 3 or split_str[-2] not in ["L", "R"]:
                warnings.warn(
                    f"Unexpected channel name: {channel}"
                    f", no hemisphere indication. Ignoring."
                )
                continue
            else:
                grouped_roi = "_".join(split_str[:-2])
        else:
            grouped_roi = "_".join(split_str[:-1])
        channel_map[channel] = grouped_roi
    return channel_map


def get_magic_channel_map(channels, combine_amy=True, combine_nac=True, whitelist=None):
    """
    Get a channel map that should magically work.
    
    Parameters
    ----------
    channels : list of str
        Names of channels
    combine_amy : bool, optional
        Whether to combine Amygdala channels
    combine_nac : bool, optional
        Whether to combine Nucleus Accumbens channels
    whitelist : None or list of str, optional
        A list specifiying a list of allowable channel prefixes.

    Returns
    -------
    channel_map : dict
        Maps individual channel names to grouped channel names
    """
    group_names = ["Cx_Cg", "Cx_IL", "Cx_PrL", "Hipp_V", "VTA", "Thal_MD"]
    if combine_amy:
        group_names.append("Amy")
    else:
        group_names += ["Amy_BLA", "Amy_CeA"]
    if combine_nac:
        group_names.append("NAc")
    else:
        group_names += ["NAc_Core", "NAc_Shell"]
    
    channel_map = {}
    for channel in channels:
        lower_channel = channel.lower()
        flag = False
        
        # Check the whitelist.
        if whitelist is not None:
            if sum(lower_channel.startswith(i) for i in whitelist) == 0:
                continue
        
        # Check the group names.
        for group_name in group_names:
            if group_name.lower() in lower_channel:
                channel_map[channel] = group_name
                flag = True
                break

        if flag:
            continue

        if "md_thal" in lower_channel:
            channel_map[channel] = "Thal_MD"
        elif "hipp" in lower_channel:
            if "hipp_d" not in lower_channel and "d_hipp" not in lower_channel:
                channel_map[channel] = "Hipp_V"
        elif "cg_cx" in lower_channel:
            channel_map[channel] = "Cx_Cg"
        elif "il_cx" in lower_channel:
            channel_map[channel] = "Cx_IL"
        elif "prl_cx" in lower_channel:
            channel_map[channel] = "Cx_PrL"
        
    return channel_map


def remove_channels(channel_map, to_remove):
    """
    Remove channels from the channel map.

    Raises
    ------
    * ``UserWarning`` if a channel to remove isn't in the channel map.

    Parameters
    ----------
    channel_map : dict
        Maps channel names to ROI names
    to_remove : list
        List of channels to remove

    Returns
    -------
    channel_map : dict
        The input channel map with deleted keys
    """
    assert isinstance(channel_map, dict)
    assert isinstance(to_remove, list)
    for channel in to_remove:
        if channel not in channel_map:
            warnings.warn(f"Channel {channel} isn't in the channel map!")
            continue
        del channel_map[channel]
    return channel_map


def remove_channels_from_lfps(lfps, chans_fn):
    """
    Remove channels specified in the CHANS file from the LFPs.

    Parameters
    ----------
    lfps : dict
        Maps ROI names to LFP waveforms
    chans_fn : str
        CHANS filename

    Returns
    -------
    lfps : dict
        Maps ROI names to LFP waveforms
    """
    to_remove = get_removed_channels_from_file(chans_fn)
    for roi in to_remove:
        if roi in lfps:
            del lfps[roi]
    return lfps


def get_removed_channels_from_file(chans_fn):
    """
    Load a list of removed channels from a file.

    Raises
    ------
    * NotImplementedError if the file format isn't supported.

    Parameters
    ----------
    chans_fn : str
        CHANS filename

    Returns
    -------
    to_remove : list of str
        List of channels to remove.
    """
    assert isinstance(chans_fn, str)
    if chans_fn.endswith(".mat"):
        # try:
        data = loadmat(chans_fn)
        # except: for old .mat files in hdf5 format...
        assert "CHANNAMES" in data, f"{chans_fn} must contain CHANNAMES!"
        assert "CHANACTIVE" in data, f"{chans_fn} must contain CHANACTIVE!"
        channel_active = data["CHANACTIVE"].flatten()
        channel_names = np.array(
            [str(i[0]) for i in data["CHANNAMES"].flatten()],
        )
        idx = np.argwhere(channel_active == 0).flatten()
        return channel_names[idx].tolist()
    else:
        raise NotImplementedError(f"Cannot load file: {chans_fn}")


def _check_lfp_channels_in_map(lfps, channel_map, strict_checking=False):
    """
    Check that every channel in ``lfps`` is in ``channel_map``.

    Raises
    ------
    * UserWarning whenever there is a channel in ``lfps`` that isn't in ``channel_map``.
      This is upgraded to an AssertionError if ``strict_checking`` is ``True``.

    Parameters
    ----------
    lfps : dict
        Maps ROI names to LFP waveforms.
    channel_map : dict
        Maps ROI names to grouped ROI names.
    """
    for channel in lfps:
        if strict_checking:
            assert channel in channel_map, f"Channel {channel} is not in channel map!"
        elif channel not in channel_map:
            warnings.warn(f"Channel {channel} is not in channel map!")


def _check_map_channels_in_lfps(lfps, channel_map, strict_checking=False):
    """
    Check that every channel in ``channel_map`` is in ``lfps``.

    Raises
    ------
    * UserWarning whenever there is a channel in ``channel_map`` that isn't in ``lfps``.
      This is upgraded to an AssertionError if ``strict_checking`` is ``True``.

    Parameters
    ----------
    lfps : dict
        Maps ROI names to LFP waveforms.
    channel_map : dict
        Maps ROI names to grouped ROI names.
    """
    for channel in channel_map:
        if strict_checking:
            assert channel in lfps, f"Channel {channel} is not in LFPs!"
        elif channel not in lfps:
            warnings.warn(f"Channel {channel} is not in LFPs!")


if __name__ == "__main__":
    pass


###
