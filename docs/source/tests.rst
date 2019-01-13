Tests
=====

Running the testes requires ``pytest``. Tests are implemented for the
non-modeling parts of twaml. That is, data handling and plotting

To run tests simply run ``pytest`` and you'll see something of this
form:

.. code-block:: none

   $ cd /path/to/twaml
   $ pytest

   ========================== test session starts =======================
   platform darwin -- Python 3.6.8, pytest-4.0.2, py-1.7.0, pluggy-0.8.0
   rootdir: /Users/ddavis/ATLAS/analysis/twaml, inifile:
   collected 14 items

   tests/test_data.py ............                                 [ 85%]
   tests/test_viz.py ..                                            [100%]

   ========================= 14 passed in 1.24 seconds ==================
