twaml.data
==========

.. toctree::
   :maxdepth: 2

.. currentmodule:: twaml.data

The ``data`` module provides a thin wrapper around working with
persistent ROOT files, persistent h5 files, and the use of transient
pandas DataFrames.


.. autoclass:: dataset
   :members:
   :show-inheritance:
   :inherited-members:

   .. automethod:: __add__

.. autofunction:: twaml.data.from_root
.. autofunction:: twaml.data.from_pytables
.. autofunction:: twaml.data.from_h5
.. autofunction:: scale_weight_sum
