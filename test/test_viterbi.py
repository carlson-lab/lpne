"""
Test the Viterbi algorithm implementation

"""
__date__ = "February 2022"

import numpy as np

import lpne


def test_viterbi_1():
    """Make sure Viterbi returns MLE with uninformative transitions."""
    T, K, N = 3, 2, 5
    transition_mat = np.ones((K, K))
    transition_mat /= np.sum(transition_mat, axis=1, keepdims=True)
    for test_num in range(N):
        emissions = np.random.rand(T, K)
        emissions /= np.sum(emissions, axis=1, keepdims=True)
        true_seq = np.argmax(emissions, axis=1)
        pred_seqs, _ = lpne.top_k_viterbi(emissions, transition_mat)
        assert np.all(true_seq == pred_seqs[0])


def test_viterbi_3():
    """Make sure Viterbi respects a forbidden transition."""
    T, K, N = 10, 3, 50
    for test_num in range(N):
        transition_mat = np.random.rand(K, K) + 1e-6
        transition_mat[0, 1] = 0.0  # Can't go from state 0 to state 1.
        transition_mat /= np.sum(transition_mat, axis=1, keepdims=True)
        emissions = np.random.rand(T, K)
        emissions /= np.sum(emissions, axis=1, keepdims=True)
        pred_seqs, _ = lpne.top_k_viterbi(emissions, transition_mat)
        for k in range(pred_seqs.shape[0]):
            for i in range(T - 1):
                assert (
                    pred_seqs[k, i] != 0 or pred_seqs[k, i + 1] != 1
                ), f"{pred_seqs[k]}"


if __name__ == "__main__":
    pass


###
