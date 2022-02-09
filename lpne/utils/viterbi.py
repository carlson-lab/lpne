"""
Viterbi algorithm

TO DO
-----
* vectorize this
"""
__date__ = "February 2021"

import numpy as np
from scipy.special import logsumexp



def viterbi(emissions, transition_mat):
    """
    Run the Viterbi algorithm to compute the MAP label sequence.

    Parameters
    ----------
    emissions : numpy.ndarray
        Shape: [windows, n_classes]
    transition_mat : numpy.ndarray
        Shape: [n_classes, n_classes]

    Returns
    -------
    seq : numpy.ndarray
        The most likely label sequence
        Shape: [windows]
    """
    # Check parameters.
    T, K = _check_params(emissions, transition_mat)

    # Calculate the stationary distribution.
    eigvals, eigvecs = np.linalg.eig(transition_mat.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    assert np.abs(eigvals[idx] - 1.0) < 1e-6
    p0 = eigvecs[:,idx] / np.sum(eigvecs[:,idx])

    # Convert to logspace.
    log_p0 = np.log(p0)
    log_transition = np.log(transition_mat)
    log_emissions = np.log(emissions)

    # Fill in first timepoint.
    t1 = np.zeros((K,T))
    t2 = np.zeros((K,T), dtype=int)
    t1[:,0] = log_p0 + log_emissions[0]

    # Fill in the rest on the forward pass.
    for j in range(1,T):
        for i in range(K):
            temp = [[
                        t1[k,j-1],
                        log_transition[k,i],
                        log_emissions[j,i],
                     ] for k in range(K)]
            temp = np.array(temp)
            temp = logsumexp(temp, axis=1)
            t1[i,j] = np.max(temp)
            t2[i,j] = np.argmax(temp)

    # Do the backwards pass.
    map_labels = np.zeros(T, dtype=int)
    map_labels[-1] = np.argmax(t1[:,-1])
    for j in reversed(range(1,T)):
        map_labels[j-1] = t2[map_labels[j],j]
    return map_labels


def sequence_log_like(seq, emissions, transition_mat):
    """
    Calculate the label sequence log likelihood under the Markov model.

    Parameters
    ----------
    seq : numpy.ndarray
        Shape: [windows]
    emissions : numpy.ndarray
        Shape: [windows,n_classes]
    transition_mat : numpy.ndarray
        Shape: [n_classes,n_classes]

    Returns
    -------
    log_like : float
        Sequence log likelihood
    """
    T, K = _check_params(emissions, transition_mat)
    assert seq.ndim == 1
    assert len(seq) == T
    # Calculate the stationary distribution.
    eigvals, eigvecs = np.linalg.eig(transition_mat.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    assert np.abs(eigvals[idx] - 1.0) < 1e-6
    p0 = eigvecs[:,idx] / np.sum(eigvecs[:,idx])
    # Convert to logspace.
    log_p0 = np.log(p0)
    log_transition = np.log(transition_mat)
    log_emissions = np.log(emissions)
    # Get observation model log like.
    log_like = log_emissions[seq].sum()
    # Add in transition model log like.
    log_like += log_transition[seq[:-1], seq[1:]].sum()
    return log_like


def _check_params(emissions, transition_mat):
    assert emissions.ndim == 2
    T, K = emissions.shape
    assert np.allclose(np.sum(emissions, axis=1), np.ones(T))
    assert transition_mat.shape == (K,K)
    assert np.allclose(np.sum(transition_mat, axis=1), np.ones(K))
    return T, K



if __name__ == '__main__':
    pass



###
