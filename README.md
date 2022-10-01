<h1 align="center">LPNE feature extraction and classification pipeline</h1>

<h2 align="center">Code for preprocessing and building models with local field potentials</h2>


<p align="center">
<a href="https://github.com/carlson-lab/lpne/blob/master/LICENSE.md"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

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
* [Python3](https://www.python.org/) (3.7+)
* [PyTorch](https://pytorch.org) 1.11+
* [Black](https://github.com/psf/black)
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
45. automatic groups?
46. Save smoothed labels
47. More agressive normalization options
