"""
An app for making LFP features.

"""
__date__ = "December 2021 - August 2022"


from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CheckboxGroup,
    ColumnDataSource,
    Select,
    Panel,
    PreText,
    Slider,
    Tabs,
    TextInput,
    MultiSelect,
    DataTable,
    TableColumn,
)
from bokeh.plotting import curdoc
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
DEFAULT_LFP_DIR = "/Users/jack/Desktop/lpne/test_data/Data/"
DEFAULT_CHANS_DIR = "/Users/jack/Desktop/lpne/test_data/CHANS/"
DEFAULT_FEATURE_DIR = "/Users/jack/Desktop/lpne/test_data/features/"
DEFAULT_DURATION = 2
NULL_SELECTION = "No selection"
LFP_SUFFIX = "_LFP.mat"

# FILTERING
LFP_LOWCUT = 0.5
LFP_HIGHCUT = 55.0
Q = 1.5

# CHANNEL MAP
DEFAULT_MAP = dict(
    channel_names=["foo", "bar"],
    roi_names=["foo", "bar"],
)


def feature_app(doc):
    """Define the app."""

    ###################
    # Channel map tab #
    ###################
    source = ColumnDataSource(DEFAULT_MAP)

    columns = [
        TableColumn(field="channel_names", title="Channel Name"),
        TableColumn(field="roi_names", title="ROI Name"),
    ]
    data_table = DataTable(
        source=source,
        columns=columns,
        width=500,
        height=400,
        editable=True,
    )

    channel_map_button = Button(label="Populate with Defaults", default_size=200)

    def channel_map_callback():
        lfp_dir = lfp_dir_input.value
        if not os.path.exists(lfp_dir):
            alert_box.text = f"LFP directory does not exist: {lfp_dir}"
            channel_map_button.button_type = "warning"
            return
        combine_hemi = 0 in hemisphere_checkbox.active
        lfp_fns = sorted([os.path.join(lfp_dir, i) for i in multi_select_1.value])
        if len(lfp_fns) == 0:
            alert_box.text = f"No LFP filenames selected!"
            channel_map_button.button_type = "warning"
            return
        all_keys = []
        for file_num in range(len(lfp_fns)):
            if not os.path.exists(lfp_fns[file_num]):
                alert_box.text = f"LFP filename does not exist: {lfp_fns[file_num]}"
                channel_map_button.button_type = "warning"
                return
            # Load LFP data.
            lfps = lpne.load_lfps(lfp_fns[file_num])
            all_keys.append(list(lfps.keys()))
        all_keys = np.unique(all_keys)
        channel_map = lpne.get_default_channel_map(
            all_keys,
            combine_hemispheres=combine_hemi,
        )
        new_data = dict(
            channel_names=list(channel_map.keys()),
            roi_names=list(channel_map.values()),
        )
        source.data = new_data
        channel_map_button.button_type = "default"
        alert_box.text = ""

    channel_map_button.on_click(channel_map_callback)

    ###################
    # Input data tab. #
    ###################

    if os.path.exists(DEFAULT_LFP_DIR):
        initial_options = _my_listdir(DEFAULT_LFP_DIR)
    else:
        initial_options = []

    multi_select_1 = MultiSelect(
        value=[],
        options=initial_options,
        title="Select LFP files:",
        height=MULTISELECT_HEIGHT,
        width=MULTISELECT_WIDTH,
    )

    def update_multi_select_1(new_options):
        multi_select_1.options = new_options

    if os.path.exists(DEFAULT_CHANS_DIR):
        initial_options = _my_listdir(DEFAULT_CHANS_DIR)
    else:
        initial_options = []

    multi_select_2 = MultiSelect(
        value=[],
        options=initial_options,
        title="Select CHANS file or files:",
        height=MULTISELECT_HEIGHT,
        width=MULTISELECT_WIDTH,
    )

    def update_multi_select_2(new_options):
        multi_select_2.options = new_options

    def lfp_dir_input_callback(attr, old, new):
        """If the directory exists, populate the file selector."""
        if os.path.exists(new):
            load_dir = new
            if load_dir[-1] == os.path.sep:
                load_dir = load_dir[:-1]
            if len(load_dir.split(os.path.sep)) > 1:
                save_dir = load_dir.split(os.path.sep)[:-1]
                save_dir.append("features")
                save_dir.append(str(window_slider.value) + "s")
                save_dir_input.value = os.path.sep + os.path.join(*save_dir)
            update_multi_select_1(_my_listdir(load_dir))
            alert_box.text = ""
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
        if os.path.exists(new):
            update_multi_select_2(_my_listdir(new))
            alert_box.text = ""
        else:
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
            save_dir.append("features")
            save_dir.append(str(window_slider.value) + "s")
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

    outlier_checkbox = CheckboxGroup(
        labels=["Remove outliers?"],
        active=[0],
    )

    hemisphere_checkbox = CheckboxGroup(
        labels=["Combine hemispheres?"],
        active=[0],
    )

    dir_spec_checkbox = CheckboxGroup(
        labels=["Calculate directed features too?"],
        active=[],
    )

    channel_map_checkbox = CheckboxGroup(
        labels=["Use default ROIs?"],
        active=[],
    )

    save_dir_input = TextInput(
        value=DEFAULT_FEATURE_DIR,
        title="Enter feature directory:",
    )

    reset_save_button = Button(label="Reset Save Button", default_size=200)

    def reset_save_button_callback():
        save_button.button_type = "default"
        save_button.label = "Save"
        alert_box.text = ""

    reset_save_button.on_click(reset_save_button_callback)

    save_button = Button(label="Save", default_size=200)

    def save_callback():
        lfp_dir = lfp_dir_input.value
        if not os.path.exists(lfp_dir):
            alert_box.text = f"LFP directory does not exist: {lfp_dir}"
            save_button.button_type = "warning"
            return
        chans_dir = chans_dir_input.value
        if not os.path.exists(chans_dir):
            alert_box.text = f"CHANS directory does not exist: {chans_dir}"
            save_button.button_type = "warning"
            return
        feature_dir = save_dir_input.value
        if not os.path.exists(feature_dir):
            alert_box.text = f"Directory does not exist: {feature_dir}"
            save_button.button_type = "warning"
            return
        window_duration = window_slider.value
        combine_hemi = 0 in hemisphere_checkbox.active
        mark_outliers = 0 in outlier_checkbox.active
        directed_spectrum = 0 in dir_spec_checkbox.active
        overwrite = 0 in overwrite_checkbox.active
        default_channel_map = 0 in channel_map_checkbox.active
        # Get the filenames.
        saved_channels = {}
        chans_fns = sorted([os.path.join(chans_dir, i) for i in multi_select_2.value])
        lfp_fns = sorted([os.path.join(lfp_dir, i) for i in multi_select_1.value])
        if len(chans_fns) > 1 and len(chans_fns) != len(lfp_fns):
            alert_box.text = f"Unequal number of CHANS and LFP files: {len(chans_fns)} {len(lfp_fns)}"
            save_button.button_type = "warning"
            return
        elif len(chans_fns) == 1:
            chans_fns = chans_fns * len(lfp_fns)
        if len(lfp_fns) == 0:
            alert_box.text = f"No LFP filenames selected!"
            save_button.button_type = "warning"
            return
        if len(chans_fns) == 0:
            alert_box.text = f"No CHANS filenames selected!"
            save_button.button_type = "warning"
            return
        for file_num in range(len(lfp_fns)):
            # Load LFP data.
            lfps = lpne.load_lfps(lfp_fns[file_num])
            # Filter LFPs.
            lfps = lpne.filter_lfps(lfps, int(fs_input.value))

            # Remove the bad channels marked in the CHANS file.
            lfps = lpne.remove_channels_from_lfps(lfps, chans_fns[file_num])

            # Mark outliers with NaNs.
            if mark_outliers:
                lfps = lpne.mark_outliers(lfps, int(fs_input.value))

                # Print outlier summary.
                msg = lpne.get_outlier_summary(
                    lfps,
                    int(fs_input.value),
                    window_duration,
                )
                print(lfp_fns[file_num])
                print(msg)

            # Get the default channel grouping.
            if default_channel_map:
                channel_map = lpne.get_default_channel_map(
                    list(lfps.keys()),
                    combine_hemispheres=combine_hemi,
                )
            else:
                channel_map = dict(
                    zip(
                        source.data["channel_names"],
                        source.data["roi_names"],
                    )
                )
            # Average channels in the same region together.
            lfps = lpne.average_channels(lfps, channel_map)
            saved_channels = {**saved_channels, **dict(zip(lfps.keys(), repeat(0)))}

            # Make features.
            features = lpne.make_features(
                lfps,
                window_duration=window_duration,
                directed_spectrum=directed_spectrum,
            )

            # Save features.
            fn = os.path.split(lfp_fns[file_num])[-1][: -len(LFP_SUFFIX)] + ".npy"
            fn = os.path.join(feature_dir, fn)
            if not overwrite and os.path.exists(fn):
                alert_box.text = f"File already exists: {fn}"
                save_button.button_type = "warning"
                return
            lpne.save_features(features, fn)
        save_button.label = "Saved"
        save_button.button_type = "success"
        alert_box.text = f"Saved channels: {list(saved_channels.keys())}"

    save_button.on_click(save_callback)

    column_1 = column(
        lfp_dir_input,
        chans_dir_input,
        fs_input,
        window_slider,
        save_dir_input,
        hemisphere_checkbox,
        outlier_checkbox,
        dir_spec_checkbox,
        overwrite_checkbox,
        channel_map_checkbox,
        save_button,
        reset_save_button,
        alert_box,
    )
    column_2 = column(multi_select_1, multi_select_2)
    tab_1 = Panel(
        child=row(column_1, column_2),
        title="Data Stuff",
    )

    tab_2 = Panel(
        child=column(data_table, channel_map_button, alert_box),
        title="ROI Definitions",
    )

    tabs = Tabs(tabs=[tab_1, tab_2])

    doc.add_root(tabs)


def _my_listdir(dir):
    if dir == "":
        return []
    return sorted([i for i in os.listdir(dir) if not i.startswith(".")])


# Run the app.
feature_app(curdoc())


###
