"""
LPNE feature pipeline

Some general info about lpne...

"""
__date__ = "July 2021"
__version__ = "0.0.1"


from .data import \
        load_lfps, \
        get_default_channel_map, \
        get_removed_channels_from_file, \
        remove_channels, \
        average_channels, \
        save_features, \
        load_features, \
        load_labels

from .make_features import make_features

from .normalize import normalize_features

from .plotting import plot_power

from .utils import \
        write_fake_labels, \
        get_lfp_filenames, \
        get_feature_filenames, \
        get_label_filenames_from_feature_filenames, \
        get_lfp_chans_filenames, \
        get_feature_label_filenames



if __name__ == '__main__':
    pass



###
