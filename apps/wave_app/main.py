"""
Turn LFP channels into wave files.

"""
__date__ = "October 2021"

from bokeh.plotting import curdoc
from bokeh.layouts import column
from bokeh.models import Button, PreText, TextInput
import os
from scipy.io import wavfile

import lpne


DEFAULT_LFP_DIR = "/Users/jack/Desktop/lpne/test_data/Data/"
DEFAULT_CHANNEL = "Hipp_D_L_02"
DEFAULT_FS = 1000


def wave_app(doc):
    file_in = TextInput(
        value=DEFAULT_LFP_DIR,
        title="Enter input LFP file (.mat):",
    )
    file_out = TextInput(
        value=DEFAULT_LFP_DIR,
        title="Enter output file (.wav):",
    )
    fs_input = TextInput(
        value=str(DEFAULT_FS),
        title="Enter samplerate (Hz):",
    )
    channel_input = TextInput(
        value=DEFAULT_CHANNEL,
        title="Channel name:",
    )

    alert_box = PreText(text="")

    save_button = Button(label="Save LFP as WAV", width=150)

    def save_callback():
        # Make sure the samplerate is valid.
        try:
            fs = int(fs_input.value)
        except ValueError:
            save_button.button_type = "warning"
            alert_box.text = f"Invalid samplerate: {fs_input.value}"
            return
        if fs <= 0:
            save_button.button_type = "warning"
            alert_box.text = f"Invalid samplerate: {fs}"
            return
        # Try loading the LFP file.
        try:
            lfps = lpne.load_lfps(file_in.value)
        except (NotImplementedError, FileNotFoundError):
            save_button.button_type = "warning"
            alert_box.text = f"LPNE cannot load file: {file_in.value}"
            return
        # Make sure the channel is there.
        if channel_input.value not in lfps:
            save_button.button_type = "warning"
            alert_box.text = (
                f"Channel {channel_input.value} is not in: {list(lfps.keys())}"
            )
            return
        if not file_out.value.endswith(".wav"):
            save_button.button_type = "warning"
            alert_box.text = "Output file doesn't end in '.wav'!"
            return
        out_dir = os.path.split(file_out.value)[0]
        if out_dir != "" and not os.path.exists(out_dir):
            save_button.button_type = "warning"
            alert_box.text = f"Save path {out_dir} doesn't exist!"
            return
        lfp = lfps[channel_input.value]
        wavfile.write(file_out.value, fs, lfp)
        save_button.label = "Saved"
        save_button.button_type = "success"
        alert_box.text = ""

    save_button.on_click(save_callback)

    column_1 = column(
        file_in,
        file_out,
        fs_input,
        channel_input,
        save_button,
        alert_box,
    )
    doc.add_root(column_1)


# Run the app.
wave_app(curdoc())


###
