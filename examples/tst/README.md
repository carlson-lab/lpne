## README of TST Example

The tail suspension test (TST) is a behavioral assay used in mice that measures response to stress [Carlson et al., 2017](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6013844/).

This example marks the open source release of a TST dataset collected by the [Collective for Psychiatric Neuroengineering](https://www.dzirasalabs.com/). The dataset, as well as other files relevant to preprocessing, can be downloaded [here](https://duke.box.com/s/kbjtsxanvpyczyxgypzu0jkr6k9cluco)

The downloaded files are organized in several subfolders.

1. Data
    Contains Local Field Potential (LFP) data that the model will train off of.
    Needed for feature generation in feature_pipeline_tst.py.

2. CHANS
    Contains channel names that correspond to what regions the LFP data was recorded from.
    Needed for feature generation in feature_pipeline_tst.py.

3. ChannelNames.xlsx
    Contains more information about channel names 

4. labels
    Contains behavioral labels for training and testing. 
    Needed for prediction in prediction_pipeline_tst.py. 
    0 -> Homecage?
    1 -> TST
    -1 -> Ignored

5. features
    The folder that the generated features are saved to if you use the feature pipeline. These features have already been generated for you and are ready for prediction.
    Needed for prediction in prediction_pipeline_tst.py.

Once you've downloaded the files, you're ready to create features from the LFPs (cross power and [directed spectrum](https://proceedings.neurips.cc/paper/2021/file/3d36c07721a0a5a96436d6c536a132ec-Paper.pdf)) and use machine learning to predict whether or not mice in the test set are undergoing the TST for a particular window.

In addition to prediction results, running prediction_pipeline_tst.py also creates pdfs of some of the factors (group of important power features) the model is using to do prediction, as well as a random set of power features, and their reconstruction from the model.
