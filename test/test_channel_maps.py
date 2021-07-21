"""
Test lpne.channel_maps functions.

"""
__date__ = "July 2021"


import numpy as np
import pytest

import lpne


FAKE_ROIS = ['foo_L_01', 'foo_R_02', 'bar_L_01', 'bar_L_02']



def test_get_default_channel_map_1():
    """Make sure `combine_hemispheres` works."""
    rois = FAKE_ROIS
    channel_map = lpne.get_default_channel_map(rois, combine_hemispheres=True)
    assert len(channel_map.keys()) == len(FAKE_ROIS)
    assert len(np.unique(list(channel_map.values()))) == 2
    rois = FAKE_ROIS
    channel_map = lpne.get_default_channel_map(rois, combine_hemispheres=False)
    assert len(channel_map.keys()) == len(FAKE_ROIS)
    assert len(np.unique(list(channel_map.values()))) == 3


def test_remove_channels():
    """Make sure lpne.to_remove removes a channel."""
    channel_map = _get_fake_channel_map()
    original_len = len(channel_map)
    to_remove = [FAKE_ROIS[0]]
    channel_map = lpne.remove_channels(channel_map, to_remove)
    assert to_remove[0] not in channel_map
    assert len(channel_map) == original_len - 1


def test_get_removed_channels_from_file():
    """To do"""
    pass


def _get_fake_rois():
    return FAKE_ROIS


def _get_fake_lfps():
    lfps = {}
    for roi in _get_fake_rois():
        lfps[roi] = np.random.randn(20)
    return lfps


def _get_fake_channel_map():
    channel_map = {}
    for roi in _get_fake_rois():
        channel_map[roi] = roi.split('_')[0]
    return channel_map



if __name__ == '__main__':
    pass



###
