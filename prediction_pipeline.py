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
__date__ = "July 2021 - July 2022"


import numpy as np
import os
import sys

import lpne
from lpne.models import FaSae, CpSae


USAGE = "Usage:\n$ python prediction_pipeline.py <experiment_directory>"
FEATURE_SUBDIR = 'features'
LABEL_SUBDIR = 'labels'
CP_SAE = False
MODEL_KWARGS = dict(
    n_iter=1000,
    reg_strength=10.0,
)



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
    feature_fns, label_fns = lpne.get_feature_label_filenames(
            feature_dir,
            label_dir,
    )

    # Write fake labels.
    lpne.write_fake_labels(feature_dir, label_dir, n_classes=3)

    # Collect all the features and labels.
    features, labels, rois = lpne.load_features_and_labels(
            feature_fns,
            label_fns,
    )

    # Define a test/train split.
    idx = int(round(0.7 * len(features)))
    train_idx = np.arange(idx)
    test_idx = np.arange(idx, len(features))
    partition = {
        'train': train_idx,
        'test': test_idx,
    }

    # Normalize and reshape the power features.
    features = lpne.normalize_features(features, partition)
    features = lpne.unsqueeze_triangular_array(features, 1)
    features = np.transpose(features, [0,3,1,2]) # [b,f,r,r]

    # Make the model.
    model = (CpSae if CP_SAE else FaSae)(**MODEL_KWARGS)

    # Make fake groups.
    groups = np.random.randint(0,2,len(labels))

    # Plot cross power in decibels for the labels, averaged over groups.
    freqs = np.arange(features.shape[1]) # NOTE: HERE
    lpne.plot_db(features, freqs, labels, groups, rois=rois)

    # Fit the model.
    print("Training model...")
    model.fit(
            features[train_idx],
            labels[train_idx],
            groups[train_idx],
            print_freq=50,
            score_freq=50,
    )
    print("Done training.\n")

    # Save the model.
    print("Saving model...")
    model.save_state(os.path.join(exp_dir, 'model_state.npy'))

    # Plot a couple factors.
    print("Plotting factors...")
    factor = model.get_factor(0)
    lpne.plot_factor(factor, rois, fn='factor_1.pdf')
    factor = model.get_factor(1)
    lpne.plot_factor(factor, rois, fn='factor_2.pdf')

    # Plot a random window and its reconstruction.
    idx = np.random.randint(len(features))
    feature = features[idx:idx+1] # [1,f,r,r]
    rec_feature = model.reconstruct(feature.reshape(1,-1)) # [1,x]
    rec_feature = rec_feature.reshape(feature.shape) # [1,f,r,r]
    both_features = np.concatenate([feature, rec_feature], axis=0) # [1,f,r,r]
    both_features = lpne.squeeze_triangular_array(both_features, dims=(2,3))
    both_features = np.transpose(both_features, [0,2,1]) # [2,r*(r+1)//2,f]
    lpne.plot_factors(
        both_features,
        rois=rois,
        fn='reconstruction.pdf',
    )

    # Make some predictions.
    print("Making predictions...")
    predictions = model.predict(features[test_idx], groups[test_idx])
    true_labels = labels[test_idx]
    print("Test labels:")
    print(true_labels)
    print("Test predictions:")
    print(predictions)

    # Confusion matrix
    confusion = lpne.confusion_matrix(true_labels, predictions)
    print("Confusion matrix:")
    print(confusion)

    # Calculate a weighted accuracy.
    weighted_acc = model.score(
            features[test_idx],
            labels[test_idx],
            groups[test_idx],
    )
    print("Weighted accuracy on test set:", weighted_acc)


###
