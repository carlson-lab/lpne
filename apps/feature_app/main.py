"""
Feature app.

TO DO
-----
* Input the channel map
"""
__date__ = "December 2021"


from bokeh.layouts import column, row
from bokeh.models import Button, CheckboxGroup, ColumnDataSource, Select, \
        Panel, PreText, Slider, Tabs, TextInput, MultiSelect
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import figure, curdoc
import numpy as np
from itertools import repeat
import os

import lpne


# App-related constants
BUTTON_SIZE = 200
BUTTON_SIZE = 150
TEXT_INPUT_WIDTH = 500
MULTISELECT_HEIGHT = 500
MULTISELECT_WIDTH = 350
DEFAULT_LFP_DIR = '/Users/jack/Desktop/lpne/test_data/Data/'
DEFAULT_CHANS_DIR = '/Users/jack/Desktop/lpne/test_data/CHANS/'
DEFAULT_FEATURE_DIR = '/Users/jack/Desktop/lpne/test_data/features/'
DEFAULT_DURATION = 2
NULL_SELECTION = "No selection"
LFP_SUFFIX = '_LFP.mat'


# FILTERING
LFP_LOWCUT = 0.5
LFP_HIGHCUT = 55.0
Q = 1.5



def feature_app(doc):
    """Define the app."""
    ###################
    # Input data tab. #
    ###################

    def lfp_dir_input_callback(attr, old, new):
        """If the directory exists, populate the file selector."""
        if os.path.exists(new):
            load_dir = new
            if load_dir[-1] == os.path.sep:
                load_dir = load_dir[:-1]
            if len(load_dir.split(os.path.sep)) > 1:
                save_dir = load_dir.split(os.path.sep)[:-1]
                save_dir.append('labels')
                save_dir.append(str(window_slider.value)+'s')
                save_dir_input.value = os.path.sep + os.path.join(*save_dir)
        else:
            alert_box.text = f"Not a valid directory: {new}"

    lfp_dir_input = TextInput(
            value=DEFAULT_LFP_DIR,
            title="Enter LFP directory:",
            width=TEXT_INPUT_WIDTH,
    )
    lfp_dir_input.on_change("value", lfp_dir_input_callback)


    def chans_dir_input_callback(attr, old, new):
        """If the directory exists, populate the file selector."""
        if not os.path.exists(new):
            alert_box.text = f"Not a valid directory: {new}"

    chans_dir_input = TextInput(
            value=DEFAULT_CHANS_DIR,
            title="Enter CHANS directory:",
            width=TEXT_INPUT_WIDTH,
    )
    chans_dir_input.on_change("value", chans_dir_input_callback)


    window_slider = Slider(
            start=1,
            end=20,
            value=DEFAULT_DURATION,
            step=1,
            title="Window duration (s)",
    )

    def window_slider_callback(attr, old, new):
        load_dir = load_dir_input.value
        if load_dir[-1] == os.path.sep:
            load_dir = load_dir[:-1]
        if os.path.exists(load_dir) and len(load_dir.split(os.path.sep)) > 1:
            save_dir = load_dir.split(os.path.sep)[:-1]
            save_dir.append('labels')
            save_dir.append(str(window_slider.value)+'s')
            save_dir_input.value = os.path.sep + os.path.join(*save_dir)

    window_slider.on_change("value", window_slider_callback)

    fs_input = TextInput(value="1000", title="Enter samplerate (Hz):")

    alert_box = PreText(text="")

    #############
    # Save tab. #
    #############
    overwrite_checkbox = CheckboxGroup(
            labels=["Overwrite existing files?"],
            active=[],
    )

    hemisphere_checkbox = CheckboxGroup(
            labels=["Combine hemispheres?"],
            active=[0],
    )

    save_dir_input = TextInput(
            value=DEFAULT_FEATURE_DIR,
            title="Enter feature directory:",
    )

    save_button = Button(label="Save", default_size=200)

    def save_callback():
        lfp_dir = lfp_dir_input.value
        if not os.path.exists(lfp_dir):
            alert_box.text = f"Directory does not exist: {lfp_dir}"
            save_button.button_type="warning"
            return
        chans_dir = chans_dir_input.value
        if not os.path.exists(chans_dir):
            alert_box.text = f"Directory does not exist: {chans_dir}"
            save_button.button_type="warning"
            return
        feature_dir = save_dir_input.value
        if not os.path.exists(feature_dir):
            alert_box.text = f"Directory does not exist: {feature_dir}"
            save_button.button_type="warning"
            return
        window_duration = window_slider.value
        combine_hemi = 0 in hemisphere_checkbox.active
        overwrite = 0 in overwrite_checkbox.active
        # Get the filenames.
        saved_channels = {}
        lfp_fns, chans_fns = lpne.get_lfp_chans_filenames(lfp_dir, chans_dir)
        for file_num in range(len(lfp_fns)):
            # Load LFP data.
            lfps = lpne.load_lfps(lfp_fns[file_num])
            # Filter LFPs.
            lfps = lpne.filter_lfps(lfps, int(fs_input.value))
            # Get the default channel grouping.
            channel_map = lpne.get_default_channel_map(
                    list(lfps.keys()),
                    combine_hemispheres=combine_hemi,
            )
            # Load the contents of a file to determine which channels to remove.
            to_remove = lpne.get_removed_channels_from_file(chans_fns[file_num])
            # Remove these channels.
            channel_map = lpne.remove_channels(channel_map, to_remove)
            # Average channels in the same region together.
            lfps = lpne.average_channels(lfps, channel_map)
            saved_channels = {**saved_channels, **dict(zip(lfps.keys(),repeat(0)))}
            # Make features.
            features = lpne.make_features(lfps, window_duration=window_duration)
            # Save features.
            fn = os.path.split(lfp_fns[file_num])[-1][:-len(LFP_SUFFIX)] + '.npy'
            fn = os.path.join(feature_dir, fn)
            if not overwrite and os.path.exists(fn):
                alert_box.text = f"File already exists: {fn}"
                save_button.button_type="warning"
                return
            lpne.save_features(features, fn)
        save_button.label = "Saved"
        save_button.button_type="success"
        alert_box.text = f"Saved channels: {list(saved_channels.keys())}"

    save_button.on_click(save_callback)

    column_1 = column(
            lfp_dir_input,
            chans_dir_input,
            fs_input,
            window_slider,
            save_dir_input,
            hemisphere_checkbox,
            overwrite_checkbox,
            save_button,
            alert_box,
    )

    doc.add_root(column_1)


# Run the app.
feature_app(curdoc())


###
