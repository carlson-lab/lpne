"""
Run the standard experimental pipeline.

Usage:
$ python script.py experiment_directory [params_file]

"""
__date__ = "October - November 2022"


import sys

from lpne import standard_pipeline


USAGE = "Usage:\n$ python script.py experiment_directory [params_file]"


if __name__ == "__main__":
    # Check the input argument.
    if len(sys.argv) not in [2, 3]:
        quit(USAGE)
    exp_dir = sys.argv[1]
    params_fn = sys.argv[2] if len(sys.argv) == 3 else None
    standard_pipeline(exp_dir, params_fn=params_fn)


if __name__ == "__main__":
    pass


###
