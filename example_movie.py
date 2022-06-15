"""
Make a movie of LFP power features.

"""
__date__ = "August 2021 - June 2022"


import os
from feature_pipeline import WINDOW_DURATION

import lpne


FN = os.path.join('test_data', 'data', 'example_LFP.mat')
FEATURE = ['power', 'dir_spec'][0]
DURATION = 25.0
WINDOW_DURATION = 5.0


if __name__ == '__main__':
    # Load LFP data.
    lfps = lpne.load_lfps(FN)

    # Get the default channel grouping.
    channel_map = lpne.get_default_channel_map(list(lfps.keys()))

    # Average channels in the same region together.
    lfps = lpne.average_channels(lfps, channel_map)

    # Make the movie.
    lpne.make_power_movie(
            lfps,
            duration=DURATION,
            window_duration=WINDOW_DURATION,
            fps=10,
            speed_factor=3,
            feature=FEATURE,
            fn='out.gif',
    )



###
