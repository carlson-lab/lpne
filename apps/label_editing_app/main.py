"""
Label editing app.

TO DO
-----
* Filter the channels before plotting.
"""
__date__ = "September 2021"


from bokeh.layouts import column, row
from bokeh.models import Button, CheckboxGroup, ColumnDataSource, Select, \
        Panel, PreText, Slider, Tabs, TextInput
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import figure, curdoc
from matplotlib.colors import to_hex
import numpy as np
import os

import lpne


# App-related constants
BUTTON_SIZE = 200
LINE_WIDTH = 1
LFP_HEIGHT = 220
LFP_WIDTH = 1000
DEFAULT_LFP_DIR = '/Users/jack/Desktop/lpne/test_data/Data/'
DEFAULT_LABEL_DIR = '/Users/jack/Desktop/lpne/test_data/labels/'
DEFAULT_EMG_NAME = "EMG_trap"
DEFAULT_LFP_NAME = "Hipp_D_L_02"
DEFAULT_DURATION = 2
LABELS = ['Wake', 'NREM', 'REM', 'Unlabeled']
COLORS = ['dodgerblue', 'mediumseagreen', 'darkorchid', 'peru']
COLORS = [to_hex(i) for i in COLORS]
LFP_TOOLS = 'pan,xwheel_zoom,reset,box_zoom' # wheel_zoom
LABEL_TOOLS = 'lasso_select,pan,reset'
NULL_SELECTION = "No selection"



def label_editing_app(doc):
    """Define the app."""
    ##################
    # LFP trace tab. #
    ##################
    lfp_source = ColumnDataSource()
    lfp_plot = figure(tools=LFP_TOOLS, height=LFP_HEIGHT, width=LFP_WIDTH)
    lfp_plot.line(
            x="lfp_time",
            y="lfp",
            source=lfp_source,
            line_width=LINE_WIDTH,
            color="slategray",
    )
    lfp_plot.yaxis[0].axis_label = 'LFP'

    emg_plot = figure(
            tools=LFP_TOOLS,
            x_range=lfp_plot.x_range,
            height=LFP_HEIGHT,
            width=LFP_WIDTH,
    )
    emg_plot.line(
            x="lfp_time",
            y="emg",
            source=lfp_source,
            line_width=LINE_WIDTH,
            color="slategray",
    )
    emg_plot.yaxis[0].axis_label = 'EMG'

    label_source = ColumnDataSource()
    label_plot = figure(
            tools=LABEL_TOOLS,
            x_range=lfp_plot.x_range,
            height=LFP_HEIGHT,
            width=LFP_WIDTH,
    )
    label_plot.scatter(
            x="label_time",
            y="label_y",
            source=label_source,
            line_width=LINE_WIDTH,
            color="color",
    )
    label_plot.xaxis[0].axis_label = 'Time (s)'
    label_plot.yaxis[0].axis_label = 'Labels'

    def callback_0():
        new_data = {**{}, **label_source.data}
        for idx in label_source.selected.indices:
            new_data['color'][idx] = COLORS[0]
        label_source.data = new_data
        label_source.selected.indices = []

    def callback_1(label_source=label_source):
        new_data = {**{}, **label_source.data}
        for idx in label_source.selected.indices:
            new_data['color'][idx] = COLORS[1]
        label_source.data = new_data
        label_source.selected.indices = []

    def callback_2(label_source=label_source):
        new_data = {**{}, **label_source.data}
        for idx in label_source.selected.indices:
            new_data['color'][idx] = COLORS[2]
        label_source.data = new_data
        label_source.selected.indices = []

    def callback_3(label_source=label_source):
        new_data = {**{}, **label_source.data}
        for idx in label_source.selected.indices:
            new_data['color'][idx] = COLORS[3]
        label_source.data = new_data
        label_source.selected.indices = []

    label_callbacks = [callback_0, callback_1, callback_2, callback_3]
    label_buttons = []
    for i in range(len(LABELS)):
        button = Button(label=LABELS[i], default_size=BUTTON_SIZE)
        button.on_click(label_callbacks[i])
        label_buttons.append(button)

    ###################
    # Input data tab. #
    ###################

    def lfp_dir_input_callback(attr, old, new):
        """If the directory exists, populate the file selector."""
        if os.path.exists(new):
            load_dir = new
            if load_dir[-1] == os.path.sep:
                load_dir = load_dir[:-1]
            lfp_select.options = _my_listdir(load_dir)
        else:
            alert_box.text = f"Not a valid directory: {new}"

    def label_dir_input_callback(attr, old, new):
        """If the directory exists, populate the file selector."""
        if os.path.exists(new):
            load_dir = new
            if load_dir[-1] == os.path.sep:
                load_dir = load_dir[:-1]
            label_select.options = _my_listdir(load_dir)
        else:
            alert_box.text = f"Not a valid directory: {new}"

    def save_fn_callback(attr, old, new):
        npy_savefile_input.value = os.path.join(
                label_dir_input.value,
                label_select.value,
        )

    # LFP stuff.
    lfp_dir_input = TextInput(
            value=DEFAULT_LFP_DIR,
            title="Enter LFP directory:",
    )
    lfp_dir_input.on_change("value", lfp_dir_input_callback)
    lfp_select = Select(
            title="Select LFP file:",
            value=NULL_SELECTION,
            options=_my_listdir(lfp_dir_input.value),
    )

    # Label stuff.
    label_dir_input = TextInput(
            value=DEFAULT_LABEL_DIR,
            title="Enter LFP directory:",
    )
    label_dir_input.on_change("value", label_dir_input_callback)
    label_select = Select(
            title="Select label file:",
            value=NULL_SELECTION,
            options=_my_listdir(label_dir_input.value),
    )
    label_select.on_change("value", save_fn_callback)

    # Parameters
    window_slider = Slider(
            start=1,
            end=20,
            value=DEFAULT_DURATION,
            step=1,
            title="Window duration (s)",
    )
    fs_input = TextInput(value="1000", title="Enter samplerate (Hz):")
    hipp_channel_input = TextInput(
            value=DEFAULT_LFP_NAME,
            title="Enter LFP channel:",
    )
    emg_channel_input = TextInput(
            value=DEFAULT_EMG_NAME,
            title="Enter EMG channel:",
    )

    # Other.
    load_button = Button(label="Load data", default_size=BUTTON_SIZE, width=100)
    alert_box = PreText(text="")

    def load_callback():
        """Load the LFPs and labels."""
        lfp_fn = lfp_select.value
        if lfp_fn == NULL_SELECTION:
            load_button.button_type = "warning"
            alert_box.text = "Select an LFP file!"
            return
        lfp_fn = os.path.join(lfp_dir_input.value, lfp_fn)
        if not os.path.isfile(lfp_fn):
            load_button.button_type = "warning"
            alert_box.text = f"LFP file doesn't exist: {lfp_fn}"
            return
        lfps = lpne.load_lfps(lfp_fn)

        # Make sure the samplerate is valid.
        try:
            fs = int(fs_input.value)
        except ValueError:
            load_button.button_type = "warning"
            alert_box.text = f"Invalid samplerate: {fs_input.value}"
            return
        if fs <= 0:
            load_button.button_type = "warning"
            alert_box.text = f"Invalid samplerate: {fs}"
            return

        # Make sure the EMG and LFP channels are valid.
        emg_channel = emg_channel_input.value
        hipp_channel = hipp_channel_input.value
        all_keys = sorted(list(lfps.keys()))
        if emg_channel not in lfps:
            load_button.button_type = "warning"
            alert_box.text = \
                    f"Didn't find the channel '{emg_channel}' in: {all_keys}"
            return
        if hipp_channel not in lfps:
            load_button.button_type = "warning"
            alert_box.text = \
                    f"Didn't find the channel '{hipp_channel}' in: {all_keys}"
            return

        # Assign the source data.
        emg_trace = lfps[emg_channel].flatten()
        lfp_trace = lfps[hipp_channel].flatten()
        lfp_times = 1/fs * np.arange(len(emg_trace))
        label_fn = label_select.value
        if label_fn == NULL_SELECTION:
            load_button.button_type = "warning"
            alert_box.text = "Select a label file!"
            return
        label_fn = os.path.join(label_dir_input.value, label_fn)
        if not os.path.isfile(label_fn):
            load_button.button_type = "warning"
            alert_box.text = f"Label file doesn't exist: {label_fn}"
            return
        try:
            labels = lpne.load_labels(label_fn)
        except ValueError:
            load_button.button_type = "warning"
            alert_box.text = f"Error loading labels: {label_fn}"
            return
        window = window_slider.value
        label_times = window * np.arange(len(labels)) + window/2
        color = [COLORS[i] for i in labels]
        new_lfp_data = dict(
            lfp_time=lfp_times,
            emg=emg_trace,
            lfp=lfp_trace,
        )
        new_label_data = dict(
            label_time=label_times,
            label_y=np.zeros(len(labels)),
            color=color,
        )
        lfp_source.data = new_lfp_data
        label_source.data = new_label_data
        load_button.button_type="success"
        load_button.label = "Loaded"

    load_button.on_click(load_callback)


    #############
    # Save tab. #
    #############

    # .npy saving...
    npy_text = PreText(text="Save labels in .npy format")
    npy_savefile_input = TextInput(value="", title="Enter filename:", width=400)
    npy_overwrite_checkbox = CheckboxGroup(
            labels=["Overwrite existing files?"],
            active=[],
    )
    npy_save_button = Button(
            label="Save Labels",
            default_size=BUTTON_SIZE,
            width=100,
    )

    def npy_save_callback():
        overwrite = 0 in npy_overwrite_checkbox.active
        labels = [COLORS.index(c) for c in label_source.data['color']]
        label_fn = npy_savefile_input.value
        try:
            lpne.save_labels(labels, label_fn, overwrite=overwrite)
        except AssertionError:
            npy_save_button.button_type="warning"
            alert_box.text = "File already exists!"
            return
        npy_save_button.label = "Saved"
        npy_save_button.button_type="success"
        alert_box.text = ""

    npy_save_button.on_click(npy_save_callback)

    # .csv saving...
    csv_text = PreText(text="Save labels in .csv format")
    csv_savefile_input = TextInput(value="", title="Enter filename:", width=400)
    csv_overwrite_checkbox = CheckboxGroup(
            labels=["Overwrite existing files?"],
            active=[],
    )
    csv_save_button = Button(
            label="Save Labels",
            default_size=BUTTON_SIZE,
            width=100,
    )

    def csv_save_callback():
        overwrite = 0 in csv_overwrite_checkbox.active
        labels = [COLORS.index(c) for c in label_source.data['color']]
        label_fn = csv_savefile_input.value
        if not overwrite and os.path.isfile(label_fn):
            csv_save_button.button_type="warning"
            alert_box.text = "File already exists!"
            return
        if not label_fn.endswith('.csv') and not label_fn.endswith('.txt'):
            csv_save_button.button_type="warning"
            alert_box.text = "CSV file should have a .csv or .txt extension."
            return
        # Format the data.
        window = window_slider.value
        t1 = window * np.arange(len(labels)) # onsets
        t2 = t1 + window # offsets
        lines = []
        idx = 0
        prev_label, start_t = None, None
        for j in range(len(t1)):
            label = labels[j]
            if label != prev_label:
                if j > 0 and prev_label != '_':
                    token = LABELS[prev_label]
                    line = f"{start_t:.6f}\t{t1[j]:.6f}\t{token}\n"
                    lines.append(line)
                start_t = t1[j]
                prev_label = label
            if j == len(t1)-1 and prev_label != '_':
                token = LABELS[prev_label]
                line = f"{start_t:.6f}\t{t1[j]:.6f}\t{token}\n"
                lines.append(line)
            idx += 1
        # Save the data.
        with open(label_fn, 'w+') as f:
            f.writelines(lines)
        csv_save_button.label = "Saved"
        csv_save_button.button_type="success"
        alert_box.text = ""

    csv_save_button.on_click(csv_save_callback)

    # Fake data.
    x = [1,2,3,4,5]
    y = [5,5,4,6,2]
    label_y = [0]*len(x)
    color = [COLORS[-1]]*len(x)
    lfp_source.data = dict(
        lfp_time=x,
        emg=y,
        lfp=y,
    )
    label_source.data = dict(
        label_time=x,
        label_y=label_y,
        color=color,
    )

    file_inputs = row(
            column(lfp_dir_input, lfp_select),
            column(label_dir_input, label_select),
    )
    file_inputs = column(
            file_inputs,
            window_slider,
            fs_input,
            hipp_channel_input,
            emg_channel_input,
            load_button,
            alert_box,
    )

    tab_1 = Panel(
            child=file_inputs,
            title="Select Files",
    )

    buttons = row(*label_buttons)
    plots = column(lfp_plot, emg_plot, label_plot, alert_box)
    tab_2 = Panel(child=column(buttons, plots), title="Edit Labels")

    npy_save_things = column(
            npy_text,
            npy_savefile_input,
            npy_overwrite_checkbox,
            npy_save_button,
    )
    csv_save_things = column(
            csv_text,
            csv_savefile_input,
            csv_overwrite_checkbox,
            csv_save_button,
    )
    save_things = column(row(npy_save_things, csv_save_things), alert_box)

    tab_3 = Panel(child=save_things, title="Save Labels")

    tabs = Tabs(tabs=[tab_1, tab_2, tab_3])
    doc.add_root(tabs)



def _my_listdir(dir):
    try:
        return [NULL_SELECTION] + \
                sorted([i for i in os.listdir(dir) if not i.startswith('.')])
    except:
        return [NULL_SELECTION]


# Run the app.
label_editing_app(curdoc())


###
