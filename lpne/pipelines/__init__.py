"""
pipelines subpackage

"""
import yaml
import warnings


# Load the default pipeline parameters.
try:
    with open("lpne/pipelines/default_params.yaml") as f:
        DEFAULT_PIPELINE_PARAMS = yaml.safe_load(f)
except:
    warnings.warn("Could not load lpne/pipelines/default_params.yaml!")
    DEFAULT_PIPELINE_PARAMS = None

from .standard_pipeline import standard_pipeline
