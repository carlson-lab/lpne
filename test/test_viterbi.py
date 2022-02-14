"""
Test the Viterbi algorithm implementation

"""
__date__ = "February 2022"

import numpy as np

import lpne



def test_viterbi_1():
    """Make sure Viterbi returns MLE with uninformative transitions."""
    T, K, N  = 3, 2, 5
    transition_mat = np.ones((K,K))
    transition_mat /= np.sum(transition_mat, axis=1, keepdims=True)
    for test_num in range(N):
        emissions = np.random.rand(T,K)
        emissions /= np.sum(emissions, axis=1, keepdims=True)
        true_seq = np.argmax(emissions, axis=1)
        pred_seq = lpne.viterbi(emissions, transition_mat)
        assert np.all(true_seq == pred_seq)


def test_viterbi_2():
    """Make sure Viterbi returns a more likely estimate than random."""
    T, K, N  = 3, 2, 5
    for test_num in range(N):
        transition_mat = np.random.rand(K,K)
        transition_mat /= np.sum(transition_mat, axis=1, keepdims=True)
        emissions = np.random.rand(T,K)
        emissions /= np.sum(emissions, axis=1, keepdims=True)
        pred_seq = lpne.viterbi(emissions, transition_mat)
        alt_seq = np.copy(pred_seq)
        idx = np.random.randint(T)
        alt_seq[idx] = (alt_seq[idx] + np.random.randint(K-1)) % K
        viterbi_log_like = lpne.sequence_log_like(
                pred_seq,
                emissions,
                transition_mat,
        )
        alt_log_like = lpne.sequence_log_like(
                alt_seq,
                emissions,
                transition_mat,
        )
        assert viterbi_log_like >= alt_log_like



if __name__ == '__main__':
    pass


###
