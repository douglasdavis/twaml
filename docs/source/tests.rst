Tests
=====

Running the testes requires ``pytest``. Tests are implemented for the
non-modeling parts of twaml. That is, data handling and plotting

To run tests simply run pytest

.. code-block:: none

    $ cd /path/to/twaml
    $ pytest
    ================================= test session starts =========================
    platform linux -- Python 3.6.6, pytest-3.10.0, py-1.7.0, pluggy-0.8.0
    rootdir: /home/ddavis/ATLAS/analysis/twaml, inifile:
    collected 7 items

    tests/test_data.py ......                                                [ 85%]
    tests/test_viz.py .                                                      [100%]

    ================================= 7 passed in 0.99 seconds ====================
