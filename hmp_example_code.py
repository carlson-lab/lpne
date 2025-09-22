"""
HMP example code

"""
__date__ = "November 2024"

import numpy as np
from scipy import integrate
from scipy.stats import hmean



def _landau_density(x, mu, sigma):
    """
    Copied from: github.com/benjaminpatrickevans/harmonicmeanp

    Computes the density of the Landau distribution. Note: This could be improved for efficiency,
    for example, using pylandau. Here I have used a naive implementation to keep dependencies small.
    """
    fn = lambda t:  np.exp(-t * ((x-mu)/sigma) - (2/np.pi) * t * np.log(t)) * np.sin(2 * t)
    return 1 / (mu * sigma) * integrate.quad(fn, 0, np.inf, limit=1000)[0]


def get_corrected_hmpval(hmpval, L, w_r=1.0):
    """
    Get the corrected harmonic mean p value.
    
    hmpval: harmonic mean p-value
    L: total number of tests
    w_r: sum of weights for the tests performed
    """
    mu = np.log(L) + 0.874
    sigma = np.pi / 2
    landau = lambda x: _landau_density(x, mu, sigma)
    res = integrate.quad(
        landau,
        w_r / hmpval,
        np.inf,
        limit=10000,
        epsabs=1e-6,
    )[0]
    return res


if __name__ == '__main__':
    n_tests = 1000
    pvals = 0.02 * np.ones(n_tests) # fill in with your p-values
    hmp = hmean(pvals)
    print("HMP:", hmp)
    corrected_hmp = get_corrected_hmpval(hmp, n_tests)
    print("Corrected HMP:", corrected_hmp)

###