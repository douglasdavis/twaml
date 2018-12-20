``twaml.data``
==============

The ``data`` module provides a thin wrapper around working with
persistent ROOT files, persistent h5 files, and the use of transient
pandas DataFrames

Classes
-------

.. currentmodule:: twaml.data

``twaml.data.dataset``
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dataset
   :members:
   :show-inheritance:
   :inherited-members:

Dataset Building Functions
--------------------------

.. currentmodule:: twaml.data


``twaml.data.root_dataset``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: root_dataset

``twaml.data.pytables_dataset``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytables_dataset

``twaml.data.h5_dataset``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: h5_dataset


Helper Functions
----------------

``twaml.data.scale_weight_sum``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: scale_weight_sum
