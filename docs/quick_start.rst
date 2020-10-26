Python Quick Start
==================

Installation
------------

Python version of napkinXC can be easily installed from PyPy repository on Linux and MacOS (Windows is currently not supported).
It requires modern C++ compiler, CMake and Git installed:

.. code:: sh

    pip install napkinxc

or directly from the GitHub repository:

.. code:: sh

    pip install pip install git+https://github.com/mwydmuch/napkinXC.git


Usage
-----

Minimal example of usage:

```
from napkinxc.datasets import load_dataset
from napkinxc.models import PLT
from napkinxc.measures import precision_at_k

X_train, Y_train = load_dataset("eurlex-4k", "train")
X_test, Y_test = load_dataset("eurlex-4k", "test")
plt = PLT("eurlex-model")
plt.fit(X_train, Y_train)
Y_pred = plt.predict(X_test, top_k=1)
print(precision_at_k(Y_test, Y_pred, k=1))
```




