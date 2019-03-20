Getting Started
===============

.. toctree::
   :maxdepth: 2

Requirements
------------

It is highy recommended to use the `Anaconda
<https://www.anaconda.com/>`_ python distribution. Most of the
required libraries (outside of the bleeding edge machine learning
packages) will be included with Anaconda environments.

The bare requirements for data handling, plotting, and testing include
(enforced by ``requirements.txt``, see file for verions):

- uproot
- pandas
- scikit-learn
- matplotlib
- h5py
- pytables
- numexpr (to ensure pandas.eval acceleration)

Since twaml is in an early development stage specific versions may
change randomly.  listed and tests are run with the latest available
installation from PyPI or Anaconda/conda-forge.

For training and testing models (not enforced by ``requirements.txt``)

- tensorflow
- pytorch
- xgboost

For building documentation

- sphinx
- sphinx_rtd_theme
- m2r


Base Setup in a venv
--------------------

A base setup without the machine learning libraries just requires a
pip installation of the ``twaml``.

.. code-block:: none

   $ python3 -m venv ~/.venvs/twaml-venv
   $ source ~/.venvs/twaml-venv/bin/activate
   $ cd /path/to/twaml
   $ pip install .

This will make the ``twaml.data`` and ``twaml.viz`` APIs accessible.


Example GPU Anaconda Setup
--------------------------

Start with a fresh Anaconda virtual environment:

.. code-block:: none

   $ conda create -n twaml python=3.7
   $ conda activate twaml
   $ conda install numpy matplotlib pandas scikit-learn pytables pytest h5py numexpr
   $ pip install uproot
   $ conda install pytorch torchvision cuda100 -c pytorch ## requires recent nvidia linux drivers
   $ conda install tensorflow-gpu ## or just tensorflow
   $ pip install .
