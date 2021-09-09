"""
Make sleep labels using the Python pipeline.

"""
__date__ = "September 2021"


import os

import lpne



if __name__ == '__main__':
    lfp_dir = 'test_data/Data/'
    lfp_fns = ['example_LFP.mat']
    lfp_fns = [os.path.join(lfp_dir, i) for i in lfp_fns]

    lpne.make_sleep_labels(lfp_fns, 'Hipp_D_L_02', 'EMG_trap', 2.0, 2.0, 1000)
