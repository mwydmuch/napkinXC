#!/usr/bin/env python3

from napkinxc.models import PLT
from napkinxc.measures import precision_at_k
from napkinxc.datasets import load_dataset

X_train, Y_train, X_test, Y_test = load_dataset("eurlex-4k")
plt = PLT(verbose=2)
plt.fit(X_train, Y_train)
Y_pred = plt.predict(X_test, top_k=1)
print(precision_at_k(Y_test, Y_pred, k=1))
