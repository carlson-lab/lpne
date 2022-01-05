"""
Project data onto a network.

Add a plot of predictions?
"""
__date__ = "January 2022"

from bokeh.plotting import curdoc
from bokeh.layouts import column
from bokeh.models import Button, PreText, TextInput, MultiSelect, \
        RadioButtonGroup
import numpy as np
import os
from sklearn.metrics import confusion_matrix

import lpne


DEFAULT_FEATURE_DIR = '/Users/jack/Desktop/lpne/test_data/features/'
DEFAULT_LABEL_DIR = '/Users/jack/Desktop/lpne/test_data/labels/'
DEFAULT_MODEL_FN = '/Users/jack/Desktop/lpne/test_data/model_state.npy'

MULTISELECT_HEIGHT = 500
MULTISELECT_WIDTH = 350
LABELS = ["CP SAE", "FA SAE"]



def project_app(doc):

    radio_button_group = RadioButtonGroup(labels=LABELS, active=0)

    if os.path.exists(DEFAULT_FEATURE_DIR):
        initial_options = _my_listdir(DEFAULT_FEATURE_DIR)
    else:
        initial_options = []

    multi_select = MultiSelect(
            value=[],
            options=initial_options,
            title="Select feature files:",
            height=MULTISELECT_HEIGHT,
            width=MULTISELECT_WIDTH,
    )

    def update_multi_select(new_options, multi_select=multi_select):
        multi_select.options = new_options


    def feature_dir_input_callback(attr, old, new):
        """If the directory exists, populate the file selector."""
        if os.path.exists(new):
            load_dir = new
            if load_dir[-1] == os.path.sep:
                load_dir = load_dir[:-1]
            update_multi_select(_my_listdir(load_dir))
        else:
            alert_box.text = f"Not a valid directory: {new}"


    model_in = TextInput(
            value=DEFAULT_MODEL_FN,
            title="Enter a model filename (.npy):",
    )
    label_dir_in = TextInput(
            value=DEFAULT_LABEL_DIR,
            title="Enter a label directory:",
    )
    feature_dir_in = TextInput(
            value=str(DEFAULT_FEATURE_DIR),
            title="Enter the feature directory:",
    )
    feature_dir_in.on_change("value", feature_dir_input_callback)

    project_button = Button(label="Project Data", width=150)

    alert_box = PreText(text="")


    def project_callback():
        # Load the features.
        feature_dir = feature_dir_in.value
        if not os.path.exists(feature_dir):
            project_button.button_type = "warning"
            alert_box.text = "Feature directory doesn't exist!"
            return
        feature_fns = sorted([os.path.join(feature_dir,i) for i in multi_select.value])
        if len(feature_fns) == 0:
            project_button.button_type = "warning"
            alert_box.text = "No feature files selected!"
            return
        label_dir = label_dir_in.value
        if not os.path.exists(label_dir):
            project_button.button_type = "warning"
            alert_box.text = "Label directory doesn't exist!"
            return
        label_fns = []
        for feature_fn in feature_fns:
            label_fn = os.path.join(label_dir, os.path.split(feature_fn)[1])
            if not os.path.exists(label_fn):
                project_button.button_type = "warning"
                alert_box.text = f"Label file doesn't exist: {label_fn}!"
                return
            label_fns.append(label_fn)
        # Load the labels.
        features, labels, rois = \
                lpne.load_features_and_labels(feature_fns, label_fns)
        # Normalize the power features.
        partition = {'train': np.arange(len(features))}
        features = lpne.normalize_features(features, partition)
        # Load the model.
        if not os.path.exists(model_in.value):
            project_button.button_type = "warning"
            alert_box.text = "Model file doesn't exist!"
            return
        if radio_button_group.active == 0:
            features = lpne.unsqueeze_triangular_array(features, 1)
            features = np.transpose(features, [0,3,1,2])
            model = lpne.CpSae()
        else:
            features = features.reshape(len(features), -1)
            model = lpne.FaSae()
        try:
            model.load_state(model_in.value)
        except (TypeError, ValueError):
            project_button.button_type = "warning"
            alert_box.text = "Incorrect model type!"
            return

        # Make predictions.
        predictions = model.predict(features)
        # Confusion matrix
        confusion = confusion_matrix(labels, predictions)

        # Calculate a weighted accuracy.
        acc = model.score(features, labels)

        message = f"Confusion matrix (rows are true labels, columns are " \
                  f"predicted labels):\n{confusion}\n\nWeighted accuracy: {acc}"

        project_button.label = "Projected"
        project_button.button_type="success"
        alert_box.text = message


    project_button.on_click(project_callback)

    column_1 = column(
            model_in,
            radio_button_group,
            label_dir_in,
            feature_dir_in,
            multi_select,
            project_button,
            alert_box,
    )
    doc.add_root(column_1)


def _my_listdir(dir):
    return sorted([i for i in os.listdir(dir) if not i.startswith('.')])


# Run the app.
project_app(curdoc())


###
