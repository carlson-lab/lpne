"""
Semi-automated sleep labeling.

"""
__date__ = "September 2021"


import matplotlib as mpl
from matplotlib.widgets import LassoSelector
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import iirnotch, lfilter, stft, butter

import lpne


ALPHA = 0.8 # 0.2

RATIO_1 = (2, 20, 55)
RATIO_2 = (2, 4.5, 9)
ORDER = 4 # bandpass
EMG_LOWCUT = 30.0
EMG_MIDDLECUT = 60.0
EMG_HIGHCUT = 249.0
LFP_LOWCUT = 0.5
LFP_HIGHCUT = 55.0
Q = 1.5



def make_sleep_labels(lfp_fns, hipp_channel, emg_channel, window_duration,
    window_step, fs, rms_limits=(None, None), emg_limits=(None, None)):
    """
    Label windowed LFPs as Wake, NREM, and REM.

    Parameters
    ----------
    lfps : dict
    hipp_channel : str
    emg_channel : str
    window_duration : float
    window_step : float
    fs : int

    Returns
    -------
    labels :

    """
    from bokeh.models import ColumnDataSource
    from bokeh.plotting import figure, output_file, show

    source = ColumnDataSource()

    fig = figure(plot_height=600, plot_width=720, tooltips=[("Title", "@title"), ("Released", "@released")])
    fig.circle(x="x", y="y", source=source, size=8, color="color", line_color=None)
    fig.xaxis.axis_label = "IMDB Rating"
    fig.yaxis.axis_label = "Rotten Tomatoes Rating"

    currMovies = [
        {'imdbid': 'tt0099878', 'title': 'Jetsons: The Movie', 'genre': 'Animation, Comedy, Family', 'released': '07/06/1990', 'imdbrating': 5.4, 'imdbvotes': 2731, 'country': 'USA', 'numericrating': 4.3, 'usermeter': 46},
        {'imdbid': 'tt0099892', 'title': 'Joe Versus the Volcano', 'genre': 'Comedy, Romance', 'released': '03/09/1990', 'imdbrating': 5.6, 'imdbvotes': 23680, 'country': 'USA', 'numericrating': 5.2, 'usermeter': 54},
        {'imdbid': 'tt0099938', 'title': 'Kindergarten Cop', 'genre': 'Action, Comedy, Crime', 'released': '12/21/1990', 'imdbrating': 5.9, 'imdbvotes': 83461, 'country': 'USA', 'numericrating': 5.1, 'usermeter': 51},
        {'imdbid': 'tt0099939', 'title': 'King of New York', 'genre': 'Crime, Thriller', 'released': '09/28/1990', 'imdbrating': 7, 'imdbvotes': 19031, 'country': 'Italy, USA, UK', 'numericrating': 6.1, 'usermeter': 79},
        {'imdbid': 'tt0099951', 'title': 'The Krays', 'genre': 'Biography, Crime, Drama', 'released': '11/09/1990', 'imdbrating': 6.7, 'imdbvotes': 4247, 'country': 'UK', 'numericrating': 6.4, 'usermeter': 82}
    ]

    source.data = dict(
        x = [d['imdbrating'] for d in currMovies],
        y = [d['numericrating'] for d in currMovies],
        color = ["#FF9900" for d in currMovies],
        title = [d['title'] for d in currMovies],
        released = [d['released'] for d in currMovies],
        imdbvotes = [d['imdbvotes'] for d in currMovies],
        genre = [d['genre'] for d in currMovies]
    )

    output_file("graph.html")
    show(fig)
    quit()

    from flask import Flask, render_template
    from bokeh.embed import components
    import webbrowser
    from threading import Timer
    import os

    from bokeh.plotting import figure
    p = figure(plot_width=400, plot_height=400)

    # add a circle renderer with a size, color, and alpha
    p.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)

    app = Flask(__name__)

    @app.route("/")

    def index():
        return "hello world"


    # https://stackoverflow.com/a/63216793
    # The reloader has not yet run - open the browser
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:8080/')

    app.run(host="127.0.0.1", port=8080, debug=True)
    quit()


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
    # X2 = np.stack([ratio_2s, ratio_1s],axis=1)

    cs = ['b']*len(X1)
    # fig, axarr = plt.subplots(ncols=2)
    ax = plt.gca()
    ax.scatter(X1[:,0], X1[:,1], c=cs, alpha=ALPHA, s=1.0)
    ax.set_xlabel("DHipp RMS Amplitude")
    ax.set_ylabel("log EMG Power")
    ax.set_xlim(rms_limits)
    ax.set_ylim(emg_limits)
    # axarr[1].scatter(X2[:,0], X2[:,1], c=cs, alpha=ALPHA, s=1.0)
    # axarr[1].set_xlabel("Ratio 2 (2,4.5,9)")
    # axarr[1].set_ylabel("Ratio 1 (2,20,55)")

    colors = ['darkorchid', 'mediumseagreen', 'r', 'b']
    labels = ['Wake', 'NREM', 'REM', '_']

    ITR = 0

    def onselect(vertices, itr={'foo':0}):
        # global ITR
        ITR = itr['foo']
        print("ITR", ITR)
        path = mpl.path.Path(vertices)
        # if SELECT_AXIS == 0:
        X = X1
        # else:
        #     X = X2
        for j in range(len(X)):
            if path.contains_point(X[j]):
                cs[j] = colors[ITR]
        ITR += 1
        itr['foo'] += 1
        print("closing in onselect!")
        plt.close()
        print("done!")

        # fig, axarr = plt.subplots(ncols=2)
        ax = plt.gca()
        ax.scatter(X1[:,0], X1[:,1], c=cs, alpha=ALPHA, s=1.0)
        ax.set_xlabel("DHipp RMS Amplitude")
        ax.set_ylabel("log EMG Power")
        ax.set_xlim(rms_limits)
        ax.set_ylim(emg_limits)
        # axarr[1].scatter(X2[:,0], X2[:,1], c=cs, alpha=ALPHA, s=1.0)
        # axarr[1].set_xlabel("Ratio 2 (2,4.5,9)")
        # axarr[1].set_ylabel("Ratio 1 (2,20,55)")

        if ITR < 3:
            plt.title(f"Select {labels[ITR]} points")
        else:
            plt.title("Close window and enter EMG threshold.")
        # lasso = LassoSelector(axarr[SELECT_AXIS], onselect)
        lasso = LassoSelector(ax, onselect)
        # plt.xlabel("2-4.5 / 2-9 Hz DHipp")
        print("show in onselect")
        fig = plt.gcf()
        plt.show()

        # try:
        #     while fig.number in plt.get_fignums():
        #         plt.pause(0.1)
        # except:
        #     plt.close(fig.number)
        #     raise
        print("end of inselect")


    plt.title(f"Select {labels[ITR]} points")
    # lasso = LassoSelector(axarr[SELECT_AXIS], onselect)
    lasso = LassoSelector(ax, onselect)
    fig = plt.gcf()
    plt.show()

    # try:
    #     while fig.number in plt.get_fignums():
    #         plt.pause(0.1)
    # except:
    #     plt.close(fig.number)
    #     raise


    thresh = input("Enter EMG treshold: ")
    thresh = float(thresh)
    temp_c = colors[labels.index('Wake')]
    for i in range(len(cs)):
        if emgs[i] > thresh:
            cs[i] = temp_c
    plt.close()

    # # fig, axarr = plt.subplots(ncols=2)
    # ax = plt.gca()
    # ax.scatter(X1[:,0], X1[:,1], c=cs, alpha=0.1, s=1.0)
    # ax.set_xlabel("DHipp RMS Amplitude")
    # ax.set_ylabel("log EMG Power")
    # ax.set_xlim(rms_limits)
    # ax.set_ylim(emg_limits)
    # # axarr[1].scatter(X2[:,0], X2[:,1], c=cs, alpha=0.1, s=1.0)
    # # axarr[1].set_xlabel("Ratio 2 (2,4.5,9)")
    # # axarr[1].set_ylabel("Ratio 1 (2,20,55)")
    # plt.title("Close this window")
    # plt.show()

    for c, label in zip(colors, labels):
        print(f"{label}: {len([i for i in cs if i==c])}")
    print("Done.")
    return






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



if __name__ == '__main__':
    pass



###
