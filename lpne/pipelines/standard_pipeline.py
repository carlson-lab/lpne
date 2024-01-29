"""
Standard experimental piepline

TODO: explicitly check for duplicate mice on different days
TODO: group the parameters differently or add kwargs to functions?
"""
__date__ = "October 2022 - January 2024"
__all__ = [
    "standard_pipeline",
]


import numpy as np
import os
import yaml

import lpne


USAGE = "Usage:\n$ python script.py <experiment_directory>"


def standard_pipeline(exp_dir, params=None, params_fn=None):
    """
    Run the standard experimental pipeline.

    If both ``params`` and ``params_fn`` are ``None``, default parameters are used.

    Parameters
    ----------
    exp_dir : str
        Experiment directory
    params : None or dict, optional
        Pipeline parameters
    params_fn : None or str, optional
        Pipeline parameters filename, relative to ``exp_dir``
    """
    assert params is None or params_fn is None, "Only set one of params and params_fn!"
    if params_fn is not None:
        # Load the parameters.
        fn = os.path.join(exp_dir, params_fn)
        try:
            with open(fn) as f:
                params = yaml.safe_load(f)
        except:
            quit(f"Cannot load {fn}!")
    elif params is None and params_fn is None:
        # Use the default parameters.
        params = lpne.DEFAULT_PIPELINE_PARAMS

    print("Parameters:\n\n" + yaml.dump(params))

    # Load feature, CHANS, label, and LFP filenames with all the checks.
    chans_fns, feature_fns, label_fns, lfp_fns = lpne.get_all_fns(
        exp_dir, **params["file"]
    )
    groups, group_map = lpne.infer_groups_from_fns(lfp_fns)
    model_fn = os.path.join(exp_dir, params["file"]["model_fn"])

    # Make the plotting directory.
    plot_dir = os.path.join(exp_dir, params["file"]["plot_subdir"])
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Make the features.
    if params["pipeline"]["make_features"]:
        # Load the channel map.
        channel_map_fn = os.path.join(exp_dir, params["file"]["channel_map_fn"])
        channel_map = lpne.load_channel_map(channel_map_fn)
        # Make the features for each filename.
        for file_num in range(len(lfp_fns)):
            print(f"File {file_num+1}/{len(lfp_fns)}:", lfp_fns[file_num])
            # Load the LFPs.
            lfps = lpne.load_lfps(lfp_fns[file_num])
            # Remove the bad channels marked in the CHANS file.
            lfps = lpne.remove_channels_from_lfps(lfps, chans_fns[file_num])
            # Filter the LFPs.
            lfps = lpne.filter_lfps(
                lfps,
                params["preprocess"]["fs"],
                lowcut=params["preprocess"]["filter_lowcut"],
                highcut=params["preprocess"]["filter_highcut"],
            )
            if params["preprocess"]["remove_outliers"]:
                # Mark outliers with NaNs.
                lfps = lpne.mark_outliers(
                    lfps,
                    params["preprocess"]["fs"],
                    lowcut=params["preprocess"]["outlier_lowcut"],
                    highcut=params["preprocess"]["filter_highcut"],
                    mad_threshold=params["preprocess"]["outlier_mad_threshold"],
                )
                # Print outlier summary.
                msg = lpne.get_outlier_summary(
                    lfps,
                    params["preprocess"]["fs"],
                    params["preprocess"]["window_duration"],
                )
                print(msg)
            # Average channels and combine outliers in the same group.
            lfps = lpne.average_channels(
                lfps,
                channel_map,
                **params["preprocess"]["channel_map_params"],
            )
            # Make features.
            features = lpne.make_features(
                lfps,
                fs=params["preprocess"]["fs"],
                min_freq=params["preprocess"]["feature_min_freq"],
                max_freq=params["preprocess"]["feature_max_freq"],
                window_duration=params["preprocess"]["window_duration"],
                window_step=params["preprocess"]["window_step"],
                max_n_windows=params["preprocess"]["max_n_windows"],
                spectral_granger=params["preprocess"]["spectral_granger"],
                directed_spectrum=params["preprocess"]["directed_spectrum"],
                csd_params=params["preprocess"]["csd_params"],
            )
            # Save the features.
            lpne.save_features(features, feature_fns[file_num])

    # Load all the features and labels.
    features, labels, rois, groups, freqs = lpne.load_features_and_labels(
        feature_fns,
        label_fns,
        group_map=group_map,
        return_freqs=True,
    )

    print("\nUnique labels:", np.unique(labels))

    # Normalize the features and reshape.
    features = lpne.normalize_features(
        features,
        mode=params["training"]["normalize_mode"],
    )  # [b,r(r+1)//2,f]
    features = lpne.unsqueeze_triangular_array(features, 1)  # [b,r,r,f]
    features = np.transpose(features, [0, 3, 1, 2])  # [b,f,r,r]

    # Make some summary plots showing the different classes and mice.
    if params["pipeline"]["summary_plots"]:
        print("Plotting summary plots...")
        for mode in ["abs", "diff"]:
            lpne.plot_db(
                features,
                freqs,
                labels,
                groups,
                rois=rois,
                mode=mode,
                fn=os.path.join(plot_dir, f"avg_label_powers_{mode}.pdf"),
            )

    # Do some cross-validation to estimate generalization and train a single model.
    print()
    model_class = lpne.get_model_class(params["training"]["model_name"])

    if params["pipeline"]["train_model"]:
        model = lpne.GridSearchCV(
            model_class(**params["training"]["model_kwargs"]),
            params["training"]["param_grid"],
            cv=params["training"]["cv"],
            test_size=params["training"]["test_size"],
            cv_seed=params["training"]["grid_search_cv_seed"],
            training_seed=params["training"]["grid_search_training_seed"],
        )
        model.fit(features, labels, groups)

        # Save the model.
        print("\nSaving...")
        model.save_state(model_fn)

        # Print out the best parameters and the score.
        print("\nModel parameters:", model.best_params_)
        print(f"Model score: {model.best_score_:.3f}")

    if params["pipeline"]["evaluate_model"]:
        # Reload the model to expose methods. TODO: simplify this!
        model = model_class(**params["training"]["model_kwargs"])
        model.load_state(model_fn)

        # Print out some statistics summarizing the reconstruction quality.
        print("\n" + lpne.get_reconstruction_summary(model, features))

        # Plot the factors.
        print("\nPlotting factors...")
        factors = np.stack([model.get_factor(i) for i in range(model.z_dim)], axis=0)
        lpne.plot_factors(
            factors,
            rois,
            fn=os.path.join(plot_dir, "factors.pdf"),
        )
        lpne.plot_factors(
            factors[: len(model.classes_)],
            rois,
            fn=os.path.join(plot_dir, "predictive_factors.pdf"),
        )


if __name__ == "__main__":
    pass


###
