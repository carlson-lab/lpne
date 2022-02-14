"""
An example label prediction script.

Required directory structure:

::

    exp_dir
    │
    ├── features
    │   ├── foo.npy
    │   ├── bar.npy
    │   └── baz.npy
    │
    └── labels
        ├── foo.npy
        ├── bar.npy
        └── baz.npy


"""
__date__ = "July 2021 - February 2022"


import numpy as np
import os
from sklearn.metrics import confusion_matrix
import sys

import lpne
from lpne.models import FaSae, CpSae


USAGE = "Usage:\n$ python prediction_pipeline.py <experiment_directory>"
FEATURE_SUBDIR = 'features'
LABEL_SUBDIR = 'labels'
CP_SAE = True
TENSORBOARD = False



if __name__ == '__main__':
    # Check input arguments.
    if len(sys.argv) != 2:
        quit(USAGE)
    exp_dir = sys.argv[1]
    assert os.path.exists(exp_dir), f"{exp_dir}"
    feature_dir = os.path.join(exp_dir, FEATURE_SUBDIR)
    assert os.path.exists(feature_dir), f"{feature_dir}"
    label_dir = os.path.join(exp_dir, LABEL_SUBDIR)
    assert os.path.exists(label_dir), f"{label_dir}"

    # Get the filenames.
    feature_fns, label_fns = \
            lpne.get_feature_label_filenames(feature_dir, label_dir)

    # Write fake labels.
    lpne.write_fake_labels(feature_dir, label_dir, n_classes=3)

    # Collect all the features and labels.
    features, labels, rois = \
            lpne.load_features_and_labels(feature_fns, label_fns)

    if CP_SAE:
        features = lpne.unsqueeze_triangular_array(features, 1)


    # Define a test/train split.
    idx = int(round(0.7 * len(features)))
    train_idx = np.arange(idx)
    test_idx = np.arange(idx, len(features))
    partition = {
        'train': train_idx,
        'test': test_idx,
    }

    # Normalize the power features.
    features = lpne.normalize_features(features, partition)
    if CP_SAE:
        features = np.transpose(features, [0,3,1,2])
    if not CP_SAE:
        features = features.reshape(len(features), -1)

    # Make the model.
    if CP_SAE:
        log_dir = 'logs/' if TENSORBOARD else None
        model = CpSae(n_iter=50, log_dir=log_dir)
    else:
        model = FaSae(n_iter=50)

    # Fit the model.
    print("Training model...")
    if CP_SAE:
        # Make fake groups.
        groups = np.random.randint(0,2,len(labels))
        model.fit(features[train_idx], labels[train_idx], groups[train_idx], print_freq=5)
    else:
        model.fit(features[train_idx], labels[train_idx], print_freq=5)
    print("Done training.\n")

    # Plot factor.
    print("Plotting factors...")
    factor = model.get_factor(0)
    lpne.plot_factor(factor, rois, fn='factor_1.pdf')
    factor = model.get_factor(1)
    lpne.plot_factor(factor, rois, fn='factor_2.pdf')

    # Save the model.
    print("Saving model...")
    model.save_state(os.path.join(exp_dir, 'model_state.npy'))

    # Make some predictions.
    print("Making predictions...")
    if CP_SAE:
        predictions = model.predict(features[test_idx], groups[test_idx])
    else:
        predictions = model.predict(features[test_idx])
    true_labels = labels[test_idx]
    print("Test labels:")
    print(true_labels)
    print("Test predictions:")
    print(predictions)

    # Confusion matrix
    confusion = confusion_matrix(true_labels, predictions)
    print("Confusion matrix:")
    print(confusion)

    # Calculate a weighted accuracy.
    if CP_SAE:
        weighted_acc = model.score(
                features[test_idx],
                labels[test_idx],
                groups[test_idx],
        )
    else:
        weighted_acc = model.score(
                features[test_idx],
                labels[test_idx],
        )
    print("Weighted accuracy on test set:", weighted_acc)


###
