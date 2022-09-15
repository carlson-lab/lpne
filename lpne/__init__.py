"""
LPNE feature pipeline

Code for preprocessing and building factor models with local field potentials.

"""
__date__ = "July 2021 - August 2022"
__version__ = "0.1.6"
try:
    with open(".git/logs/HEAD", "r") as fh:
        __commit__ = fh.read().split('\n')[-2]
except:
	__commit__ = "unknown commit"

INVALID_LABEL = -1
INVALID_GROUP = -1

from .models import FaSae, CpSae, GridSearchCV, dcsfa_nmf

from .plotting import \
        plot_db, \
        plot_lfps, \
        plot_factor, \
        plot_factors, \
        plot_power, \
        plot_spec, \
        make_power_movie

from .preprocess.channel_maps import \
        IGNORED_KEYS, \
        average_channels, \
        get_excel_channel_map, \
        get_default_channel_map, \
        remove_channels, \
        remove_channels_from_lfps, \
        get_removed_channels_from_file

from .preprocess.directed_spectrum import get_directed_spectrum

from .preprocess.filter import filter_signal, filter_lfps

from .preprocess.make_features import make_features

from .preprocess.normalize import normalize_features, normalize_lfps

from .preprocess.outlier_detection import mark_outliers

from .utils.data import \
        load_lfps, \
        save_features, \
        load_features, \
        save_labels, \
        load_labels, \
        load_features_and_labels

from .utils.utils import \
        write_fake_labels, \
        get_lfp_filenames, \
        get_feature_filenames, \
        get_label_filenames_from_feature_filenames, \
        get_lfp_chans_filenames, \
        get_feature_label_filenames, \
        get_weights, \
        unsqueeze_triangular_array, \
        squeeze_triangular_array, \
        get_outlier_summary, \
        flatten_dir_spec_features, \
        confusion_matrix

from .utils.viterbi import top_k_viterbi, get_label_stats



if __name__ == '__main__':
    pass



###
