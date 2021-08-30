"""
LPNE feature pipeline

Some general info about lpne...

"""
__date__ = "July - August 2021"
__version__ = "0.0.1"


from .preprocess.channel_maps import \
        average_channels, \
        get_default_channel_map, \
        remove_channels, \
        get_removed_channels_from_file

from .preprocess.make_features import make_features

from .preprocess.normalize import normalize_features

from .models import FaSae

from .plotting import \
        plot_factor, \
        plot_power, \
        make_power_movie

from .utils.utils import \
        write_fake_labels, \
        get_lfp_filenames, \
        get_feature_filenames, \
        get_label_filenames_from_feature_filenames, \
        get_lfp_chans_filenames, \
        get_feature_label_filenames

from .utils.data import \
        load_lfps, \
        save_features, \
        load_features, \
        save_labels, \
        load_labels



if __name__ == '__main__':
    pass



###
