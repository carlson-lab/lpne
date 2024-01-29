"""
Test the Wilson factorization implementation

"""
__date__ = "January 2024"

import numpy as np
from scipy.signal import csd


from lpne.preprocess.directed_measures import _wilson_factorize


def test_wilson_1():
    """Check that the factorization works."""
    for nperseg in [4, 5]:
        # Make random data.
        d = np.random.randn(3, 99)
        _, S = csd(
            d[:, None],
            d[None],
            scaling="spectrum",
            nperseg=nperseg,
            return_onesided=False,
        )

        tol = 1e-6
        max_iter = 1000
        S = np.transpose(S, (2, 0, 1))  # [f,r,r]
        # Factorize.
        H, Z = _wilson_factorize(S[None], max_iter, tol)
        # Check the factorization worked.
        Z = Z[0]
        assert np.allclose(Z, Z.T)
        H = np.transpose(H[0], (1, 2, 0))
        rec_S = np.einsum("ijf,jk,lkf->fil", H, Z, H.conj())
        assert np.allclose(S, rec_S)


if __name__ == "__main__":
    pass


###
