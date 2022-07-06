.. lpne documentation master file, created by
   sphinx-quickstart on Wed Jul 14 12:15:06 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to lpne's documentation!
================================

**Installation**

.. code-block:: bash

   $ git clone https://github.com/carlson-lab/lpne.git
   $ cd lpne
   $ pip install .
   $ pytest test # run tests
   $ cd docs
   $ make html # build docs

.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. automodule:: lpne
  :members:


  .. toctree::
     :maxdepth: 1
     :caption: Docs:

     lpne.models
     lpne.plotting
     lpne.preprocess
     lpne.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
