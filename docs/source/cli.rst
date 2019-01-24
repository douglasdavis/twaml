Command Line Applications
=========================

.. toctree::
   :maxdepth: 1

.. currentmodule:: twaml.clapps

``twaml-root2pytables``
-----------------------

Convert a set of ROOT files into a single pytables HDF5 file.

.. command-output:: twaml-root2pytables --help

An example that uses the default ``--tree-name`` and
``--weight-name``, while only saving the branches ``b1`` and ``b2``
and requiring the branch ``elmu`` to be true to save the event.

.. code-block:: none

   $ twaml-root2pytables -i file.root -o file.h5 --branches b1 b2 --true-branches elmu

The docs for the function that is being called:

.. autofunction:: root2pytables
