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
__date__ = "July 2021"


import numpy as np
import os
import sys

import lpne
from lpne.models import FaSae


USAGE = "Usage:\n$ python prediction_pipeline.py <experiment_directory>"
FEATURE_SUBDIR = 'features'
LABEL_SUBDIR = 'labels'



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
    lpne.write_fake_labels(feature_dir, label_dir)

    # Collect all the features and labels.
    features, labels = [], []
    for feature_fn, label_fn in zip(feature_fns, label_fns):
        temp = lpne.load_features(feature_fn)
        rois = temp['rois']
        features.append(temp['power'])
        labels.append(lpne.load_labels(label_fn))
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    n_classes = 1 + np.max(labels)
    n_features = features.shape[1] * features.shape[2]
    print("Number of features:", n_features)

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
    features = features.reshape(len(features), -1)

    # Calculate class weights.
    n = len(train_idx)
    class_counts = [len(np.argwhere(labels[train_idx]==i).flatten()) for i in range(n_classes)]
    print("Class counts:", class_counts)
    class_weights = n / (n_classes * np.array(class_counts))
    print("Class weights:", class_weights)

    # Make the model.
    model = FaSae(
        n_features,
        n_classes,
        class_weights=class_weights,
        reg_strength=1.0,
        weight_reg=0.0,
        nonnegative=True,
    )

    # Fit the model.
    print("Training model...")
    model.fit(features[train_idx], labels[train_idx])
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
    predictions = model.predict(features[test_idx])
    print("Test labels:")
    true_labels = labels[test_idx]
    print(true_labels)
    print("Test predictions:")
    print(predictions)

    # Confusion matrix
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(n_classes):
        idx_1 = np.argwhere(true_labels == i).flatten()
        for j in range(n_classes):
            idx_2 = np.argwhere(predictions == j).flatten()
            confusion[i,j] = len(np.intersect1d(idx_1, idx_2))
    print("Confusion matrix:")
    print(confusion)

    # Calculate a weighted accuracy.
    weighted_acc = model.score(
            features[test_idx],
            labels[test_idx],
            class_weights,
    )
    print("Weighted accuracy on test set:", weighted_acc)


###
