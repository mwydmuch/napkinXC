Python Quick Start
==================

Installation
------------

Python (3.8+) version of napkinXC can be easily installed from PyPy repository on Linux, MacOS and Windows.
It requires modern C++17 compiler, CMake and Git installed:

.. code:: sh

    pip install napkinxc

or directly from the GitHub repository:

.. code:: sh

    pip install pip install git+https://github.com/mwydmuch/napkinXC.git


Usage
-----

napkinxc module contains three submodules: 
- models that contains all the model classes 
- datasets that contains functions for loading datasets
- measures that contains functions for evaluating models


Minimal example of usage:

.. code:: python

    from napkinxc.datasets import load_dataset
    from napkinxc.models import PLT
    from napkinxc.metrics import precision_at_k

    X_train, Y_train = load_dataset("eurlex-4k", "train")
    X_test, Y_test = load_dataset("eurlex-4k", "test")
    plt = PLT("eurlex-model")
    plt.fit(X_train, Y_train)
    Y_pred = plt.predict(X_test, top_k=1)
    print(precision_at_k(Y_test, Y_pred, k=1))





