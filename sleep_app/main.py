"""
Automated sleep labeling app.

TO DO
-----
* clean imports
* restructure
* two plots
* check out subsampling
"""
__date__ = "September 2021"

from bokeh.layouts import column
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc

# more imports
from bokeh.models.callbacks import CustomJS
from bokeh.io import show, output_notebook
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Button, TextInput, MultiSelect, \
        Panel, Tabs, Slider, PreText, CheckboxGroup
from bokeh.plotting import figure
from bokeh.themes import Theme
from matplotlib.colors import to_hex
import numpy as np
import os
import numpy as np
from scipy.signal import iirnotch, lfilter, stft, butter

import lpne


# App-related constants
BUTTON_SIZE = 200
MULTISELECT_HEIGHT = 500
SCATTER_SIZE = 8
DEFAULT_LFP_DIR = '/Users/jack/Desktop/lpne/test_data/Data/'
TOOLS = 'lasso_select,pan,wheel_zoom,reset'
WAKE_COLOR = to_hex('dodgerblue')
NREM_COLOR = to_hex('mediumseagreen')
REM_COLOR = to_hex('darkorchid')
UNLABELED_COLOR = to_hex('peru')

# Preprocessing constants
RATIO_1 = (2, 20, 55)
RATIO_2 = (2, 4.5, 9)
ORDER = 4 # bandpass
EMG_LOWCUT = 30.0
EMG_MIDDLECUT = 60.0
EMG_HIGHCUT = 249.0
LFP_LOWCUT = 0.5
LFP_HIGHCUT = 55.0
Q = 1.5


def my_app(doc):
    """Sleep labeling app."""
    source = ColumnDataSource()
    plot = figure(tools=TOOLS)
    plot.circle(
            x="x",
            y="y",
            source=source,
            size=SCATTER_SIZE,
            color="color",
            line_color=None,
    )

    wake_button = Button(label="Wake", default_size=BUTTON_SIZE)
    nrem_button = Button(label="NREM", default_size=BUTTON_SIZE)
    rem_button = Button(label="REM", default_size=BUTTON_SIZE)
    unlabeled_button = Button(label="Unlabeled", default_size=BUTTON_SIZE)

    wake_button.js_on_click(
        CustomJS(
            args=dict(source=source),
            code=f"""
                var color = source.data['color']
                var idx = source.selected.indices;
                var selected_length = idx.length;
                for (var i=0; i<selected_length; i++) {{
                    color[idx[i]] = '{WAKE_COLOR}';
                }}
                source.selected.indices = [];
                source.change.emit();
            """,
    ))
    nrem_button.js_on_click(
        CustomJS(
            args=dict(source=source),
            code=f"""
                var color = source.data['color']
                var idx = source.selected.indices;
                var selected_length = idx.length;
                for (var i=0; i<selected_length; i++) {{
                    color[idx[i]] = '{NREM_COLOR}';
                }}
                source.selected.indices = [];
                source.change.emit();
            """,
    ))
    rem_button.js_on_click(
        CustomJS(
            args=dict(source=source),
            code=f"""
                var color = source.data['color']
                var idx = source.selected.indices;
                var selected_length = idx.length;
                for (var i=0; i<selected_length; i++) {{
                    color[idx[i]] = '{REM_COLOR}';
                }}
                source.selected.indices = [];
                source.change.emit();
            """,
    ))
    unlabeled_button.js_on_click(
        CustomJS(
            args=dict(source=source),
            code=f"""
                var color = source.data['color']
                var idx = source.selected.indices;
                var selected_length = idx.length;
                for (var i=0; i<selected_length; i++) {{
                    color[idx[i]] = '{UNLABELED_COLOR}';
                }}
                source.selected.indices = [];
                source.change.emit();
            """,
    ))


    def load_dir_input_callback(attr, old, new):
        """
        If the directory exists, populate the file selector.

        """
        if os.path.exists(new):
            update_multi_select(os.listdir(new))
        else:
            print(attr, "invalid")

    multi_select = MultiSelect(
            value=[],
            options=os.listdir(DEFAULT_LFP_DIR),
            title="Select LFP files:",
            height=MULTISELECT_HEIGHT,
    )

    def update_multi_select(new_options, multi_select=multi_select):
        multi_select.options = new_options
        print("updated!")

    load_dir_input = TextInput(value=DEFAULT_LFP_DIR, title="Enter LFP directory:")
    load_dir_input.on_change("value", load_dir_input_callback)

    emg_channel_input = TextInput(value="EMG_trap", title="Enter EMG channel name:")
    hipp_channel_input = TextInput(value="Hipp_D_L_02", title="Enter Hipp channel name:")
    window_slider = Slider(
            start=1,
            end=20,
            value=2,
            step=1,
            title="Window duration (s)",
    )
    fs_input = TextInput(value="1000", title="Enter samplerate (Hz):")

    alert_box = PreText(text="")

    load_button = Button(label="Load LFPs", default_size=BUTTON_SIZE, width=100)


    def load_callback(load_dir_input=load_dir_input, \
        multi_select=multi_select, emg_channel_input=emg_channel_input, \
        hipp_channel_input=hipp_channel_input, window_slider=window_slider, \
        fs_input=fs_input, load_button=load_button, alert_box=alert_box, \
        source=source):
        """
        Load the LFPs.


        """
        # Make sure the LFP files are selected.
        lfp_dir = load_dir_input.value
        lfp_fns = sorted([os.path.join(lfp_dir,i) for i in multi_select.value])
        if len(lfp_fns) == 0:
            load_button.button_type = "warning"
            alert_box.text = "No LFP files selected!"
            return
        # Make sure the EMG and Hipp channel names are selected.
        emg_channel = emg_channel_input.value
        hipp_channel = hipp_channel_input.value
        if len(emg_channel) == 0 or len(hipp_channel) == 0:
            load_button.button_type = "warning"
            alert_box.text = "Specify the EMG and Hippocampus channel names!"
            return

        # Just load the first LFP to make sure the right channels exist.
        lfps = lpne.load_lfps(lfp_fns[0])
        all_keys = list(lfps.keys())
        if emg_channel not in lfps:
            load_button.button_type = "warning"
            alert_box.text = f"Didn't find the channel '{emg_channel}' in: {all_keys}"
            return
        if hipp_channel not in lfps:
            load_button.button_type = "warning"
            alert_box.text = f"Didn't find the channel '{emg_channel}' in: {all_keys}"
            return

        # Make sure the samplerate is valid.
        try:
            fs = int(fs_input.value)
        except ValueError:
            load_button.button_type = "warning"
            alert_box.text = f"Invalid samplerate: {fs_input.value}"
        if fs <= 0:
            load_button.button_type = "warning"
            alert_box.text = f"Invalid samplerate: {fs}"

        # Load and process the rest of the LFPs.
        load_button.label = 'Loading...'
        X1 = get_emg_lfp_features(
                lfp_fns,
                hipp_channel,
                emg_channel,
                window_slider.value,
                window_slider.value,
                fs,
        )

        # Update source.
        new_data = dict(
            x=X1[:,0],
            y=X1[:,1],
            color=[UNLABELED_COLOR]*len(X1),
        )
        source.data = new_data

        load_button.label = "Loaded"
        load_button.button_type="success"
        msg = f"Successfully loaded LFPs.\n\nFound {len(X1)} windows."
        alert_box.text = msg

    load_button.on_click(load_callback)

    # Save tab.
    save_dir_input = TextInput(value=DEFAULT_LFP_DIR, title="Enter label directory:")
    save_button = Button(label="Save", default_size=200)

    def save_callback(save_button=save_button):
        save_button.label = "Saved"
        save_button.button_type="success"

    save_button.on_click(save_callback)

    overwrite_checkbox = CheckboxGroup(
            labels=["Overwrite existing files?"],
            active=[],
    )

    # Fake data.
    x = [1,2,3,4,5]
    y = [5,5,4,6,2]
    color = [UNLABELED_COLOR]*len(x)
    source.data = dict(
        x=x,
        y=y,
        color=color,
    )

    # Layout.
    column_1 = column(load_button, load_dir_input, multi_select)
    column_2 = column(
            emg_channel_input,
            hipp_channel_input,
            window_slider,
            fs_input,
            alert_box,
    )
    tab_1 = Panel(
        child=row(column_1, column_2),
        title="Data Stuff",
    )
    buttons = column(
            wake_button,
            nrem_button,
            rem_button,
            unlabeled_button,
    )
    tab_2 = Panel(
        child=row(buttons, plot),
        title="Sleep Selection",
    )
    tab_3 = Panel(
        child=column(save_dir_input,overwrite_checkbox,save_button),
        title="Save Labels",
    )
    tabs = Tabs(tabs=[tab_1, tab_2, tab_3])
    doc.add_root(tabs)



def get_emg_lfp_features(lfp_fns, hipp_channel, emg_channel, window_duration,
    window_step, fs):
    """
    ...

    """
    # Collect stats for each window.
    window_samples = int(fs * window_duration)
    step_samples = int(fs * window_step)
    all_emg_power, all_dhipp_rms = [], []
    for lfp_fn in lfp_fns:
        # Load the LFPs.
        lfps = lpne.load_lfps(lfp_fn)
        assert emg_channel in lfps, f"{emg_channel} not in " \
                f"{list(lfps.keys())} in file {lfp_fn}"
        assert hipp_channel in lfps, f"{hipp_channel} not in " \
                f"{list(lfps.keys())} in file {lfp_fn}"
        emg_tr = lfps[emg_channel].flatten()
        dhipp_tr = lfps[hipp_channel].flatten()
        # Calculate features of the LFP and EMG.
        emg_power = _process_emg_trace(
                emg_tr,
                fs,
                window_samples,
                step_samples,
        )
        # dhipp_ratio_1 = _process_lfp_trace(dhipp_tr[:], r=RATIO_1)
        # dhipp_ratio_2 = _process_lfp_trace(dhipp_tr[:], r=RATIO_2)
        dhipp_rms = _rms_lfp_trace(dhipp_tr, fs, window_samples, step_samples)
        all_emg_power.append(emg_power)
        all_dhipp_rms.append(dhipp_rms)
    emgs = np.concatenate(all_emg_power, axis=0)
    # ratio_1s = np.concatenate(ratio_1s, axis=0)
    # ratio_2s = np.concatenate(ratio_2s, axis=0)
    rmss = np.concatenate(all_dhipp_rms, axis=0)
    X1 = np.stack([rmss, emgs],axis=1)
    return X1


def _process_emg_trace(trace, fs, window_samples, step_samples):
	"""Calculate the RMS power over two fixed frequency bins."""
	# Subsample.
	trace = trace[::2]
	# Bandpass.
	trace = _butter_bandpass_filter(
            trace,
            EMG_LOWCUT,
            EMG_HIGHCUT,
            fs,
            order=ORDER,
    )
	# Remove electrical noise.
	for freq in range(60,int(EMG_HIGHCUT),60):
		b, a = iirnotch(freq, Q, fs)
		trace = lfilter(b, a, trace)
	b, a = iirnotch(200.0, Q, fs)
	trace = lfilter(b, a, trace)
	# Walk through the trace, collecting powers in different frequency ranges.
	data = []
	for i in range(0, len(trace)-window_samples, step_samples):
		chunk = trace[i:i+window_samples]
		f, t, spec = stft(chunk, fs=fs, nperseg=window_samples)
		spec = np.abs(spec)
		spec = np.mean(spec, axis=1) # average over the three time bins
		i1, i2, i3 = np.searchsorted(f, (EMG_LOWCUT,EMG_MIDDLECUT,EMG_HIGHCUT))
		temp = np.sqrt(np.mean(spec[i1:i2]) + np.mean(spec[i2:i3]))
		data.append(temp)
	return np.array(data)


# def _process_lfp_trace(trace, r=(2.0,4.5,9.0)):
# 	# Subsample.
# 	trace = trace[::2]
# 	# Walk through the trace, collecting powers in different frequency ranges.
# 	data = []
# 	for i in range(0, len(trace)-WINDOW_SAMPLES, STEP_SAMPLES):
# 		chunk = trace[i:i+WINDOW_SAMPLES]
# 		f, t, spec = stft(chunk, fs=FS, nperseg=WINDOW_SAMPLES)
# 		spec = np.abs(spec)
# 		spec = np.mean(spec, axis=1) # average over the three time bins
# 		i1 = np.argmin(np.abs(f - r[0]))
# 		i2 = np.argmin(np.abs(f - r[1]))
# 		i3 = np.argmin(np.abs(f - r[2]))
# 		data.append(np.sum(spec[i1:i2+1]) / np.sum(spec[i1:i3+1]))
# 	return np.array(data)


def _rms_lfp_trace(trace, fs, window_samples, step_samples):
	# Subsample.
	trace = trace[::2]
	# Bandpass.
	trace = _butter_bandpass_filter(trace, LFP_LOWCUT, LFP_HIGHCUT, fs)
	# Walk through the trace, collecting powers in different frequency ranges.
	data = []
	for i in range(0, len(trace)-window_samples, step_samples):
		chunk = trace[i:i+window_samples]
		chunk -= np.mean(chunk)
		data.append(np.sqrt(np.mean(chunk**2)))
	return np.array(data)


def _butter_bandpass(lowcut, highcut, fs, order=ORDER):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a


def _butter_bandpass_filter(data, lowcut, highcut, fs, order=ORDER):
	b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y






# Run the app.
my_app(curdoc())


###
