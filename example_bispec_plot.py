"""
Make some example bispectrum plots.

"""
__date__ = "March 2023"

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

import lpne

if __name__ == '__main__':
    lfps = lpne.load_lfps("test_data/data/example_LFP.mat")
    # lfp = lfps['Amy_CeA_L_01']
    lfp = lfps['NAc_Core_L_02']
    # lfp = lfps['Cx_PrL_R_01']


    s1 = lpne.filter_signal(lfp[:], fs=1000, lowcut=1/2, highcut=3)
    s2 = lpne.filter_signal(lfp[:], fs=1000, lowcut=21, highcut=23)
    s3 = lpne.filter_signal(lfp[:], fs=1000, lowcut=23.5, highcut=26)

    h1 = hilbert(s1)
    h2 = hilbert(s2)
    h3 = hilbert(s3)
    amp1 = np.abs(h1)
    amp2 = np.abs(h2)
    amp3 = np.abs(h3)
    angles1 = np.angle(h1)
    # angles2 = np.angle(h2)

    plt.subplot(131)
    plt.scatter(angles1, amp2, alpha=0.2, s=1.0)
    plt.subplot(132)
    plt.scatter(angles1, amp3, alpha=0.2, s=1.0)
    plt.subplot(133)
    plt.scatter(amp2, amp3, alpha=0.2, s=1.0)

    # diffs = [(b-a) % (2*np.pi) for a,b in zip(angles1, angles2)]

    # local_max = np.argwhere(amp2[1:-1] > amp2[:-2]).flatten()
    # local_max = np.intersect1d(local_max, np.argwhere(amp2[1:-1] > amp2[2:]).flatten())
    # plt.scatter(angles1[1+local_max], amp2[1+local_max], alpha=0.6)

    # plt.scatter(angles1, amp2, s=1.0, alpha=0.6)
    plt.savefig('temp.pdf')
    quit()

    # plt.scatter(angles1, angles2, alpha=0.6)
    plt.subplot(311)
    plt.hist(diffs, bins=20)

    plt.subplot(312)
    plt.hist(angles1 % (2*np.pi),bins=20)

    plt.subplot(313)
    plt.hist(angles2 % (2*np.pi),bins=20)
    plt.savefig('temp.pdf')
    quit()

    plt.plot(300 + s1[:10000])
    # plt.plot(s2[:10000])
    # plt.savefig('temp.pdf')



###