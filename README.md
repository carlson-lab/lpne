## LPNE feature extraction and classification pipeline

Code for preprocessing and building models with local field potentials.

See `feature_pipeline.py` and `prediction_pipeline.py` for usage.

#### Installation

```bash
$ git clone https://github.com/carlson-lab/lpne.git
$ cd lpne
$ pip install .
$ pytest test
```

#### Dependencies
* [Python3](https://www.python.org/)
* [MoviePy](https://github.com/Zulko/moviepy)
* [PyTorch](https://pytorch.org)
* [TensorBoard](https://github.com/tensorflow/tensorboard) (optional)


### TO DO
6. consolidate plot_factor.py and plot_power.py
7. add docstrings for channel maps
9. Add a Tucker decomposition model
10. PoE?
12. Manually remove regions from LFPs
21. Make some pre-zipped features and labels
23. Model factor movie
25. Add feature/model metadata: `datetime.datetime.now()`
26. Add random seeds
27. Make CpSae and FaSae compatible
28. Save label names with labels?
29. Plot multiple factors
30. Save the model name with the model.
31. Movie app
32. `from joblib import Parallel, delayed`
33. PCGM for tensor models
