.. napkinXC documentation master file

Welcome to napkinXC's documentation!
====================================

.. note:: Documentation is currently a work in progress!

napkinXC is an extremely simple and fast library for extreme multi-class and multi-label classification
that implements the following methods both in Python and C++:

* Probabilistic Label Trees (PLTs) - for multi-label log-time training and prediction,
* Hierarchical softmax (HSM) - for multi-class log-time training and prediction,
* Binary Relevance (BR) - multi-label baseline,
* One Versus Rest (OVR) - multi-class baseline.

All the methods decompose multi-class and multi-label into the set of binary learning problems.


Right now, the detailed descirption of methods and their parameters can be found in this paper:
`Probabilistic Label Trees for Extreme Multi-label Classification <https://arxiv.org/pdf/2009.11218.pdf>`_



.. toctree::
    :maxdepth: 1
    :caption: Contents:

    quick_start
    exe_usage
    python_api


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
