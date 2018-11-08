Getting Started
===============

Requirements
------------

It is highy recommended to use the `Anaconda
<https://www.anaconda.com/>`_ python distribution.

The bare requirements for data handling, plotting, and testing include:

- numpy
- uproot
- pandas
- scikit-learn
- matplotlib
- h5py
- pytables
- pytest

Since twaml is in an early development stage specific versions are not
listed and tests are run with the latest available installation from
PyPI or Anaconda/conda-forge.

For training and testing models

- tensorflow
- pytorch
- xgboost

For building documentation

- sphinx
- sphinx_rtd_theme
- m2r


Quick Anaconda Setup
--------------------

Start with a fresh Anaconda virtual environment:

.. code-block:: none

   $ conda create -n twaml python=3.6
   $ conda config --add channel conda-forge
   $ conda activate twaml
   $ conda install numpy matplotlib pandas scikit-learn pytables pytest h5py
   $ pip install uproot
   $ conda install tensorflow-gpu ## or just tensorflow
   $ conda install pytorch torchvision cuda92 -c pytorch
