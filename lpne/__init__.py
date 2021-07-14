"""
LPNE feature pipeline

Some general info about lpne...

"""
__date__ = "July 2021"
__version__ = "0.0.1"


from .data import \
        load_data, \
        get_default_channel_map, \
        get_removed_channels_from_file, \
        remove_channels, \
        average_channels, \
        save_features


from .make_features import make_features


from .plotting import plot_power



if __name__ == '__main__':
    pass



###
