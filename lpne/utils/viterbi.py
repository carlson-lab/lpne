"""
Viterbi algorithm

Adapted from:
https://gist.github.com/PetrochukM/afaa3613a99a8e7213d2efdd02ae4762

"""
__date__ = "February - July 2022"

import numpy as np
import torch


MIN_LOG_EMISSION = -50.0


def top_k_viterbi(emissions, transition_mat, top_k=10):
    """
    A top-K Viterbi decoder.

    Parameters
    ----------
    emissions : numpy.ndarray
        Shape: ``[windows, n_classes]``
    transition_mat : numpy.ndarray
        Shape: ``[n_classes, n_classes]``
    top_k : int, optional
        Number of paths to return. Defaults to ``10``.

    Returns
    -------
    viterbi_paths : numpy.ndarray
        Note that the returned value of k is
        ``min(top_k, n_classes ** windows)``.
        Shape: ``[k, windows]``
    viterbi_scores : numpy.ndarray
        Shape: ``[k]``
    """
    # Handle NaNs in the emissions.
    emissions[np.isnan(emissions)] = 1 / emissions.shape[1]
    # Convert to logspace and torch Tensors.
    tag_sequence = torch.tensor(np.maximum(np.log(emissions), MIN_LOG_EMISSION))
    transition_matrix = torch.tensor(np.log(transition_mat))
    sequence_length, num_tags = list(tag_sequence.size())
    path_scores = []
    path_indices = []
    # At the beginning, the maximum number of permutations is 1; therefore,
    # we unsqueeze(0) to allow for 1 permutation.
    path_scores.append(tag_sequence[0, :].unsqueeze(0))
    # Evaluate the scores for all possible paths.
    for t in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[t - 1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)
        # Best pairwise potential path score from the previous timestep.
        max_k = min(summed_potentials.size()[0], top_k)
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)
        scores = tag_sequence[t, :] + scores
        path_scores.append(scores)
        path_indices.append(paths.squeeze())
    # Construct the most likely sequence backwards.
    path_scores = path_scores[-1].view(-1)
    max_k = min(path_scores.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores, k=max_k, dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for t in reversed(path_indices):
            viterbi_path.append(int(t.view(-1)[viterbi_path[-1]]))
        # Reverse the backward path.
        viterbi_path.reverse()
        # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we
        # need to modulo.
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)
    return np.array(viterbi_paths), viterbi_scores.numpy()


def get_label_stats(viterbi_paths, viterbi_scores, n_classes):
    """
    Get statistics of the label sequences.

    This is made more robust by using top-K Viterbi decoding.

    Parameters
    ----------
    viterbi_paths : numpy.ndarray
        Shape: ``[k, windows]`` or ``[windows]``
    viterbi_scores : numpy.ndarray
        Shape: ``[k]``
    n_classes : int

    Returns
    -------
    avg_bout_counts : numpy.ndarray
        Shape: ``[n_classes]``
    avg_bout_duration : numpy.ndarray
        Shape: ``[n_classes]``
    avg_transition_counts : numpy.ndarray
        Shape: ``[n_classes, n_classes]``
    """
    if viterbi_paths.ndim == 1:
        viterbi_paths = viterbi_paths.reshape(1, -1)
        viterbi_scores = np.zeros(1)
    # Calculate stats for each label sequence.
    bout_counts, bout_durations, trans_counts = [], [], []
    for k in range(viterbi_paths.shape[0]):
        counts, durations = _bout_info(viterbi_paths[k], n_classes)
        bout_counts.append(counts)
        bout_durations.append(durations)
        transitions = _transition_info(viterbi_paths[k], n_classes)
    # Average the results.
    scores = np.exp(viterbi_scores - np.max(viterbi_scores))
    scores /= np.sum(scores)
    scores = scores.reshape(-1, 1)
    bout_counts = (np.vstack(bout_counts) * scores).sum(axis=0)
    bout_durations = (np.vstack(bout_durations) * scores).sum(axis=0)
    scores = scores.reshape(-1, 1, 1)
    transitions = (np.vstack(transitions) * scores).sum(axis=0)
    # Return stats.
    return bout_counts, bout_durations, transitions


def _bout_info(seq, n_classes):
    """Return bout counts and bout average duration."""
    bout_counts = np.zeros(n_classes)
    bout_duration = np.zeros(n_classes)
    # First window
    bout_counts[seq[0]] = 1.0
    bout_duration[seq[0]] = 1.0
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            bout_counts[seq[i]] += 1.0
        bout_duration[seq[i]] += 1.0
    return bout_counts, bout_duration / bout_counts


def _transition_info(seq, n_classes):
    """Return the transition count matrix."""
    mat = np.zeros((n_classes, n_classes))
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            mat[seq[i - 1], seq[i]] += 1.0
    return mat


if __name__ == "__main__":
    pass


###
