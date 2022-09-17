"""
Estimate the state transition times.

"""
__date__ = "September 2022"


import numpy as np
import warnings

from .utils import unsqueeze_triangular_array
from .viterbi import top_k_viterbi
from ..preprocess.make_features import make_features
from ..preprocess.normalize import normalize_features



def estimate_transition_times(lfps, fs, window_duration, transition_mat, model,
    state_from, state_to, timestep, make_feature_params={}):
    """
    Estimate the state transition times.
    
    Parameters
    ----------
    lfps : dict
        Maps region names to LFP waveforms.
    fs : int
        LFP  samplerate
    window_duration : float
    transition_mat : str or numpy.ndarray
        The transition matrix. If a ``str`` is given, this is assumed to be
        the transition matrix filename.
    model : str or lpne.BaseModel
        The trained model used to predict labels. If a ``str`` is given, this
        is assumed to be a model filename.
    state_from : int
        Defines the transition to look for.
    state_to : int
        Defines the transition to look for.
    timestep : int
        Determines how finely we want to estimate the transition times.

    Returns
    -------
    transition_times : numpy.ndarray
        The estimated transition times.
        Shape: [n]
    """
    warnings.warn(
        "lpne.estimate_transition_times is experimental and may change in " \
        "future versions!",
    )
    if isinstance(transition_mat, str):
        transition_mat = np.load(transition_mat)
    if isinstance(model, str):
        raise NotImplementedError
    
    # Make features for the LFPs.
    features = make_features(
            lfps,
            fs=fs,
            window_duration=window_duration,
            **make_feature_params,
    )['power']

    # Normalize the features.
    features = normalize_features(features)
    
    # Run per-window predictions using the model.
    features = unsqueeze_triangular_array(features, 1) # [n,r,r,f]
    features = np.transpose(features, [0,3,1,2]) # [n,f,r,r]
    proba = model.predict_proba(features) # [n,k]
    
    # Run Viterbi on the predicted labels to determine rough transition times.
    viterbi_path, _ = top_k_viterbi(proba, transition_mat, top_k=1) # [1,n]
    viterbi_path = viterbi_path.flatten() # [n]
    idx_1 = np.argwhere(viterbi_path[:-1] == state_from).flatten()
    idx_2 = np.argwhere(viterbi_path[1:] == state_to).flatten()
    t_idx = np.intersect1d(idx_1, idx_2)
    if len(t_idx) == 0:
        wanrings.warn(f"No transitions from {state_from} to {state_to} found!")
        return np.array([])
    est_times = np.zeros(len(t_idx))

    # For each transition, make features with overlap to get a better estimate.
    window_factor = int(round(window_duration / timestep))
    assert window_factor > 0, f"{window_factor} >= 0"
    window_step = window_duration / window_factor
    pinv = np.stack(
            [np.ones(window_factor+1), np.arange(window_factor+1)],
            axis=1,
    )
    pinv = np.linalg.pinv(pinv)

    # Fine tune the transition times.
    for i in range(len(t_idx)):
        idx_1 = int(fs * window_duration * t_idx[i])
        idx_2 = idx_1 + int(2 * fs * window_duration)
        temp_lfps = {}
        for key in lfps:
            temp_lfps[key] = lfps[key][idx_1:idx_2]
        features = make_features(
            temp_lfps,
            fs=fs,
            window_duration=window_duration,
            window_step=window_step,
            **make_feature_params,
        )['power']
        assert len(features) == window_factor + 1, \
                f"{len(features)} != {window_factor} + 1"

        # Normalize the features.
        features = normalize_features(features)

        # Predict and smooth.
        features = unsqueeze_triangular_array(features, 1) # [n,r,r,f]
        features = np.transpose(features, [0,3,1,2]) # [n,f,r,r]
        proba = model.predict_proba(features) # [n,k]
        proba = proba[:,np.array([state_from,state_to])] # [n,2]
        proba /= np.sum(proba, axis=1, keepdims=True) + 1e-6 # [n,2]
        betas = (pinv @ proba[:,:1]).flatten()
        if betas[1] == 0.0:
            intercept = 0.0
        else:
            intercept = (0.5 - betas[0]) / betas[1]
        est_time = window_duration * (t_idx[i] + 1)
        est_time += window_duration * intercept / (window_factor + 1)
        est_times[i] = est_time

    # Return all the collected times.
    return est_times



if __name__ == '__main__':
    pass


###