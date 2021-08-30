"""
Test lpne.make_features functions.

"""
__date__ = "July 2021"


import numpy as np

import lpne



def test_make_features_1():
    """Make sure the features have the correct shapes."""
    # Define constants.
    n_samples = 1000
    fs = 1000
    window_duration = 1.0
    for n_rois in [1,2,5]:
        # Make fake LFPs.
        lfps = {}
        for i in range(n_rois):
            lfps[f'roi_{i}'] = np.random.randn(n_samples)
        # Make features.
        res = lpne.make_features(
                lfps,
                fs=fs,
                window_duration=window_duration,
        )
        assert 'power' in res
        assert 'freq' in res
        assert 'rois' in res
        n_window = int(np.ceil(window_duration * n_samples / fs))
        roi_pairs = (n_rois * (n_rois+1)) // 2
        assert res['power'].shape[:2] == (n_window,roi_pairs), \
                f"{res['power'].shape}, ({n_window},{roi_pairs})"
        assert len(res['freq'] == res['power'].shape[2]), \
                "Inconsistent numbers of frequencies!"
        assert len(res['rois']) == n_rois, "Incorrect number of ROIs!"



if __name__ == '__main__':
    pass



###
