"""
LPNE pipeline

Code for preprocessing and building factor models with local field potentials.

"""
__date__ = "July 2021 - January 2024"
__version__ = "0.1.22"

try:
    with open(".git/logs/HEAD", "r") as fh:
        __commit__ = fh.read().split("\n")[-2]
except:
    __commit__ = "unknown commit"

INVALID_LABEL = -1
INVALID_GROUP = -1
MATLAB_IGNORED_KEYS = [
    "__header__",
    "__version__",
    "__globals__",
]
"""Ignored keys in Matlab files"""

from .models import (
    FaSae,
    CpSae,
    GridSearchCV,
    DcsfaNmf,
    get_model_class,
    get_reconstruction_stats,
    get_reconstruction_summary,
)

from .pipelines import DEFAULT_PIPELINE_PARAMS, standard_pipeline

from .plotting import (
    circle_plot,
    make_power_movie,
    plot_bispec,
    plot_db,
    plot_lfps,
    plot_factor,
    plot_factors,
    plot_power,
    plot_spec,
    simplex_plot,
)

from .preprocess.bispectrum import (
    bispectral_power_decomposition,
    get_bicoherence,
    get_bispectrum,
)
from .preprocess.channel_maps import *
from .preprocess.directed_measures import get_directed_spectral_measures
from .preprocess.filter import filter_signal, filter_lfps
from .preprocess.make_features import make_features
from .preprocess.normalize import normalize_features, normalize_lfps
from .preprocess.outlier_detection import mark_outliers
from .preprocess.phase_slope_index import get_psi

from .utils.array_utils import *
from .utils.data import *
from .utils.file_utils import *
from .utils.utils import *
from .utils.viterbi import *


if __name__ == "__main__":
    pass


###
