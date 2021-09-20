## LPNE feature extraction and classification pipeline

Code for preprocessing and building models with local field potentials.

See `feature_pipeline.py` and `prediction_pipeline.py` for usage.

#### Installation

```bash
$ git clone https://github.com/carlson-lab/lpne.git
$ cd lpne
$ pip install .
```

#### Dependencies
* [Python3](https://www.python.org/)
* [MoviePy](https://github.com/Zulko/moviepy)
* [PyTorch](https://pytorch.org)


### TO DO
1. add normalization options
2. n networks with n-class prediction?
4. remove bad windows
5. more unit tests
6. consolidate plot_factor.py and plot_power.py
7. add docstrings for channel maps
9. Add a Tucker decomposition model? <- could be a good way to separate
   between-mouse differences
10. PoE?
12. Manually remove regions from LFPs
16. Filter before normalizing channels
