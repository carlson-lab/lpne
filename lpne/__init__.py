"""
LPNE feature pipeline

Code for preprocessing and building models with local field potentials.

"""
__date__ = "July - November 2021"
__version__ = "0.0.4"



from .contrib import make_sleep_labels

from .models import FaSae

from .plotting import \
        plot_lfps, \
        plot_factor, \
        plot_power, \
        make_power_movie

from .preprocess.channel_maps import \
        IGNORED_KEYS, \
        average_channels, \
        get_default_channel_map, \
        remove_channels, \
        get_removed_channels_from_file

from .preprocess.filter import filter_signal, filter_lfps

from .preprocess.make_features import make_features

from .preprocess.normalize import normalize_features, normalize_lfps

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
        get_weights



if __name__ == '__main__':
    pass



###
