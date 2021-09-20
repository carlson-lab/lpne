"""
Automated sleep labeling app.

TO DO
-----
* two plots
* check CHANS files
"""
__date__ = "September 2021"


from bokeh.layouts import column, row
from bokeh.models import Button, CheckboxGroup, ColumnDataSource, MultiSelect, \
        Panel, PreText, Slider, Tabs, TextInput
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import figure, curdoc
from matplotlib.colors import to_hex
import numpy as np
import os
from scipy.signal import iirnotch, lfilter, stft, butter
from sklearn.neighbors import NearestNeighbors

import lpne


# App-related constants
BUTTON_SIZE = 200
MULTISELECT_HEIGHT = 500
MULTISELECT_WIDTH = 350
SCATTER_SIZE = 8
DEFAULT_LFP_DIR = '/Users/jack/Desktop/lpne/test_data/Data/'
DEFAULT_LABEL_DIR = '/Users/jack/Desktop/lpne/test_data/labels/2s/'
DEFAULT_EMG_NAME = "EMG_trap"
DEFAULT_LFP_NAME = "Hipp_D_L_02"
# DEFAULT_EMG_NAME = "Trapezius_EM"
# DEFAULT_LFP_NAME = "D_Hipp_02"
# DEFAULT_LFP_DIR = '/home/jg420/r_drive/Internal/Network/testData/'
# DEFAULT_LABEL_DIR = '/home/jg420/r_drive/Internal/Network/labels/2s/'
TOOLS = 'lasso_select,pan,wheel_zoom,reset,box_zoom'
LABELS = ['Wake', 'NREM', 'REM', 'Unlabeled']
COLORS = ['dodgerblue', 'mediumseagreen', 'darkorchid', 'peru']
COLORS = [to_hex(i) for i in COLORS]
MAX_N_POINTS = 5000

# State.
LFP_INFO = {}
COORD = None # coordinates
INVALID_FILE = None
SCATTER_SIZE = 2
ALPHA = 0.2

# File-related constants.
LFP_SUFFIX = '_LFP.mat'
LABEL_SUFFIX = '.npy'

# Preprocessing constants
DEFAULT_DURATION = 2
RATIO_1 = (2, 20, 55)
RATIO_2 = (2, 4.5, 9)
ORDER = 4 # bandpass
EMG_LOWCUT = 30.0
EMG_MIDDLECUT = 60.0
EMG_HIGHCUT = 249.0
LFP_LOWCUT = 0.5
LFP_HIGHCUT = 55.0
Q = 1.5



def sleep_app(doc):
    """Sleep labeling app."""
    ################
    # Scatter tab. #
    ################
    source = ColumnDataSource()
    plot = figure(tools=TOOLS)
    plot.scatter(
            x="x",
            y="y",
            source=source,
            size=SCATTER_SIZE,
            color="color",
            fill_alpha=ALPHA,
    )


    def callback_0():
        new_data = {**{}, **source.data}
        for idx in source.selected.indices:
            new_data['color'][idx] = COLORS[0]
        source.data = new_data
        source.selected.indices = []

    def callback_1():
        new_data = {**{}, **source.data}
        for idx in source.selected.indices:
            new_data['color'][idx] = COLORS[1]
        source.data = new_data
        source.selected.indices = []

    def callback_2():
        new_data = {**{}, **source.data}
        for idx in source.selected.indices:
            new_data['color'][idx] = COLORS[2]
        source.data = new_data
        source.selected.indices = []

    def callback_3():
        new_data = {**{}, **source.data}
        for idx in source.selected.indices:
            new_data['color'][idx] = COLORS[3]
        source.data = new_data
        source.selected.indices = []

    label_callbacks = [callback_0, callback_1, callback_2, callback_3]

    label_buttons = []
    for i in range(len(LABELS)):
        button = Button(label=LABELS[i], default_size=BUTTON_SIZE)
        button.on_click(label_callbacks[i])
        label_buttons.append(button)

    alpha_input = TextInput(
            value=str(ALPHA),
            title="Transparency:",
    )

    def alpha_callback(attr, old, new):
        global ALPHA
        try:
            ALPHA = max(min(1,float(new)),0)
            plot.scatter(
                    x="x",
                    y="y",
                    source=source,
                    size=SCATTER_SIZE,
                    color="color",
                    fill_alpha=ALPHA,
            )
        except ValueError:
            pass
        alpha_input.value = str(ALPHA)

    alpha_input.on_change("value", alpha_callback)

    size_input = TextInput(
            value=str(SCATTER_SIZE),
            title="Scatter size:",
    )

    def scatter_size_callback(attr, old, new):
        global SCATTER_SIZE
        try:
            SCATTER_SIZE = max(float(new),0)
            plot.scatter(
                    x="x",
                    y="y",
                    source=source,
                    size=SCATTER_SIZE,
                    color="color",
                    fill_alpha=ALPHA,
            )
        except ValueError:
            pass
        size_input.value = str(SCATTER_SIZE)

    size_input.on_change("value", scatter_size_callback)

    ###################
    # Input data tab. #
    ###################

    def load_dir_input_callback(attr, old, new):
        """If the directory exists, populate the file selector."""
        if os.path.exists(new):
            load_dir = new
            if load_dir[-1] == os.path.sep:
                load_dir = load_dir[:-1]
            update_multi_select(_my_listdir(load_dir))
            if len(load_dir.split(os.path.sep)) > 1:
                save_dir = load_dir.split(os.path.sep)[:-1]
                save_dir.append('labels')
                save_dir.append(str(window_slider.value)+'s')
                save_dir_input.value = os.path.sep + os.path.join(*save_dir)
        else:
            alert_box.text = f"Not a valid directory: {new}"

    if os.path.exists(DEFAULT_LFP_DIR):
        initial_options = _my_listdir(DEFAULT_LFP_DIR)
    else:
        initial_options = []

    multi_select = MultiSelect(
            value=[],
            options=initial_options,
            title="Select LFP files:",
            height=MULTISELECT_HEIGHT,
            width=MULTISELECT_WIDTH,
    )

    def update_multi_select(new_options, multi_select=multi_select):
        multi_select.options = new_options

    load_dir_input = TextInput(
            value=DEFAULT_LFP_DIR,
            title="Enter LFP directory:",
    )
    load_dir_input.on_change("value", load_dir_input_callback)

    emg_channel_input = TextInput(
            value=DEFAULT_EMG_NAME,
            title="Enter EMG channel name:",
    )
    hipp_channel_input = TextInput(
            value=DEFAULT_LFP_NAME,
            title="Enter Hipp channel name:",
    )
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

    load_button = Button(label="Load LFPs", default_size=BUTTON_SIZE, width=100)


    def load_callback(load_dir_input=load_dir_input, \
        multi_select=multi_select, emg_channel_input=emg_channel_input, \
        hipp_channel_input=hipp_channel_input, window_slider=window_slider, \
        fs_input=fs_input, load_button=load_button, alert_box=alert_box, \
        source=source):
        """Load the LFPs."""
        global INVALID_FILE, COORD, PERM
        alert_box.text = "Loading LFPs..."
        load_button.button_type = "warning"
        load_button.label = 'Loading...'
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
        try:
            lfps = lpne.load_lfps(lfp_fns[0])
        except NotImplementedError:
            load_button.button_type = "warning"
            alert_box.text = f"LPNE cannot load file: {lfp_fns[0]}"
            return
        all_keys = list(lfps.keys())
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
        try:
            COORD = get_emg_lfp_features(
                lfp_fns,
                hipp_channel,
                emg_channel,
                window_slider.value,
                window_slider.value,
                fs,
            )
        except NotImplementedError:
            load_button.button_type = "warning"
            alert_box.text = f"LPNE cannot load file: {INVALID_FILE}"
            return

        # Update source.
        PERM = np.random.permutation(len(COORD))[:MAX_N_POINTS]
        new_data = dict(
            x=COORD[PERM,0],
            y=COORD[PERM,1],
            color=[COLORS[-1]]*len(PERM),
        )
        source.data = new_data

        load_button.label = "Loaded"
        load_button.button_type="success"
        msg = f"Successfully loaded LFPs.\n\nFound {len(COORD)} windows."
        alert_box.text = msg

    load_button.on_click(load_callback)

    #############
    # Save tab. #
    #############
    overwrite_checkbox = CheckboxGroup(
            labels=["Overwrite existing files?"],
            active=[],
    )
    save_dir_input = TextInput(
            value=DEFAULT_LABEL_DIR,
            title="Enter label directory:",
    )
    save_button = Button(label="Save", default_size=200)

    def save_callback(save_button=save_button, save_dir_input=save_dir_input,
        overwrite_checkbox=overwrite_checkbox):
        """Save the sleep labels."""
        global COORD, PERM
        save_dir = save_dir_input.value
        overwrite = 0 in overwrite_checkbox.active
        if not os.path.exists(save_dir):
            alert_box.text = f"Directory does not exist: {save_dir}"
            save_button.button_type="warning"
            return
        # Standardize for better nearest neighbors.
        COORD[:,0] /= np.std(COORD[:,0])
        COORD[:,1] /= np.std(COORD[:,1])
        # Fit the nearest neighbors.
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(COORD[PERM])
        all_neighbors = neigh.kneighbors(COORD, return_distance=False).flatten()

        i = 0
        for lfp_fn in sorted(list(LFP_INFO.keys())):
            label_fn = os.path.split(lfp_fn)[-1][:-len(LFP_SUFFIX)]
            label_fn = os.path.join(save_dir, label_fn + LABEL_SUFFIX)
            j = i + LFP_INFO[lfp_fn]
            idx = all_neighbors[i:j]
            labels = [COLORS.index(source.data['color'][k]) for k in idx]
            try:
                lpne.save_labels(labels, label_fn, overwrite=overwrite)
            except AssertionError:
                save_button.button_type="warning"
                alert_box.text = "File already exists!"
                return
        save_button.label = "Saved"
        save_button.button_type="success"
        alert_box.text = ""

    save_button.on_click(save_callback)

    # Fake data.
    x = [1,2,3,4,5]
    y = [5,5,4,6,2]
    color = [COLORS[-1]]*len(x)
    source.data = dict(
        x=x,
        y=y,
        color=color,
    )

    ###########
    # Layout. #
    ###########
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
    buttons = column(*(label_buttons+[alpha_input, size_input, alert_box]))
    tab_2 = Panel(
        child=row(buttons, plot),
        title="Sleep Selection",
    )
    save_column = column(save_dir_input,overwrite_checkbox,save_button)
    tab_3 = Panel(
        child=row(save_column, alert_box),
        title="Save Labels",
    )
    tabs = Tabs(tabs=[tab_1, tab_2, tab_3])
    doc.add_root(tabs)



def get_emg_lfp_features(lfp_fns, hipp_channel, emg_channel, window_duration,
    window_step, fs):
    """ """
    global LFP_INFO, INVALID_FILE
    LFP_INFO = {}
    # Collect stats for each window.
    window_samples = int(fs * window_duration)
    step_samples = int(fs * window_step)
    all_emg_power, all_dhipp_rms = [], []
    for lfp_fn in sorted(lfp_fns):
        # assert lfp_fn.endswith(LFP_SUFFIX), \
        #         f"{lfp_fn} doesn't end with {LFP_SUFFIX}"
        # Load the LFPs.
        try:
            lfps = lpne.load_lfps(lfp_fn)
        except NotImplementedError:
            INVALID_FILE = lfp_fn
            raise NotImplementedError
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
        assert len(emg_power) == len(dhipp_rms)
        LFP_INFO[lfp_fn] = len(dhipp_rms)
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


def _my_listdir(dir):
    return sorted([i for i in os.listdir(dir) if not i.startswith('.')])



# Run the app.
sleep_app(curdoc())


###
