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

.. autofunction:: from_root
.. autofunction:: from_pytables
.. autofunction:: from_h5
.. autofunction:: scale_weight_sum
