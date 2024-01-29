"""
Define the default parameters for the pipeline.

"""
__date__ = "February 2023 - January 2024"
__all__ = ["DEFAULT_PIPELINE_PARAMS"]

import yaml


DEFAULT_PIPELINE_PARAMS_STR = """
  file:
    channel_map_fn: channel_map.csv
    chans_subdir: CHANS
    chans_suffix: _CHANS.mat
    feature_subdir: features
    label_subdir: labels
    label_suffix: .npy
    lfp_subdir: Data
    lfp_suffix: _LFP.mat
    model_fn: model_state.npy
    plot_subdir: plots
    strict_checking: false
  pipeline:
    make_features: true
    summary_plots: true
    train_model: true
    evaluate_model: true
  preprocess:
    channel_map_params:
      assert_onto: true
      check_lfp_channels_in_map: true
      check_map_channels_in_lfps: false
      strict_checking: true
    csd_params:
      detrend: constant
      window: hann
      nperseg: 512
      noverlap: 256
      nfft: null
    spectral_granger: false
    directed_spectrum: false
    feature_min_freq: 0.0
    feature_max_freq: 55.0
    filter_highcut: 55.0
    filter_lowcut: 0.5
    fs: 1000
    max_n_windows: null
    outlier_lowcut: 30.0
    outlier_mad_threshold: 15.0
    remove_outliers: true
    window_duration: 1
    window_step: null
  training:
    cv: 2 # 3
    grid_search_cv_seed: 42
    grid_search_training_seed: 42
    model_kwargs:
      batch_size: 256
      device: auto
      encoder_type: linear
      lr: 0.001
      n_iter: 50 # 1000
      rec_loss_type: lad
      reg_strength: 1.0
      z_dim: 16
    model_name: cp_sae
    normalize_mode: median
    score_mode: weighted_acc
    test_size: 2
    param_grid:
      reg_strength: [0.001, 0.01, 0.1]
  """

DEFAULT_PIPELINE_PARAMS = yaml.safe_load(DEFAULT_PIPELINE_PARAMS_STR)


if __name__ == "__main__":
    pass


###
