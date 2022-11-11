"""
Make a movie of LFP power features.

"""
__date__ = "August 2021 - November 2022"


import os

import lpne


FN = os.path.join("test_data", "data", "example_LFP.mat")
MODEL_FN = os.path.join("test_data", "model_state.npy")
FEATURE = ["power", "dir_spec"][0]
DURATION = 25.0
WINDOW_DURATION = 5.0
RECONSTRUCTION = False
MODE = "circle"
CP_SAE = False
CIRCLE_PARAMS = dict(
    freq_ticks=[0,20,40],
    min_max_quantiles=[0.6,0.99],
)


if __name__ == "__main__":
    # Load LFP data.
    lfps = lpne.load_lfps(FN)

    # Get the default channel grouping.
    channel_map = lpne.get_default_channel_map(list(lfps.keys()))

    # Average channels in the same region together.
    lfps = lpne.average_channels(lfps, channel_map)

    # Make the model.
    if RECONSTRUCTION:
        model = lpne.CpSae() if CP_SAE else lpne.FaSae()
        model.load_state(MODEL_FN)
    else:
        model = None

    # Make the movie.
    lpne.make_power_movie(
        lfps,
        duration=DURATION,
        window_duration=WINDOW_DURATION,
        fps=10,
        speed_factor=3,
        feature=FEATURE,
        mode=MODE,
        model=model,
        circle_params=CIRCLE_PARAMS,
        fn="out.webm",
    )


###
