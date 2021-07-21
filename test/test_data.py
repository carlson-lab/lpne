"""
Test lpne.data functions.

"""
__date__ = "July 2021"


import pytest

import lpne



def test_load_lfps_1():
    """Make sure lpne.load_lfp throws an error for non-existing filename."""
    with pytest.raises(FileNotFoundError):
        x = lpne.load_lfps('not_a_real_lfp_filename.mat')


def test_load_lfps_2():
    """Make sure lpne.load_lfp throws an error for non-supported file type."""
    with pytest.raises(NotImplementedError):
        x = lpne.load_lfps('possibly_a_real_lfp_filename.npy')



if __name__ == '__main__':
    pass



###
