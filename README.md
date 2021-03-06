## LPNE feature extraction and classification pipeline

Code for preprocessing and building models with local field potentials.

See `feature_pipeline.py` and `prediction_pipeline.py` for usage.

<p align="center">
<img align="middle" src="example_cpsd.gif" width="600" height="600" />
</p>

#### Installation

```bash
$ git clone https://github.com/carlson-lab/lpne.git
$ cd lpne
$ pip install .
$ pytest test # run tests
$ cd docs
$ make html # build docs
```

Then see `docs/build/html/index.html` for the docs.

#### Dependencies
* [Python3](https://www.python.org/)
* [PyTorch](https://pytorch.org) 1.11+
* [MoviePy](https://github.com/Zulko/moviepy) (optional)


### TO DO
9. Add a Tucker decomposition model
10. PoE?
12. Manually remove regions from LFPs
21. Make some pre-zipped features and labels
26. Add random seeds?
31. Movie app
32. `from joblib import Parallel, delayed`
34. mouse-specific intercepts
36. SMC for sampling label sequence posterior
37. Mouse-specific normalization options
41. Group Lasso?
42. Early stopping in `GridSearchCV`
43. ``[b,f,r,r]`` vs ``[b,r,r,f]`` shapes
44. test dB plots
45. automatic groups?
46. Save smoothed labels
