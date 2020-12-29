#!/usr/bin/env python3

# This demo shows how to train, store and later load the napkinXC model.

from napkinxc.datasets import load_dataset
from napkinxc.models import PLT
from napkinxc.measures import precision_at_k

# The beginning is the same as in the basic.py example.

# Use load_dataset function to load one of the benchmark datasets
# from XML Repository (http://manikvarma.org/downloads/XC/XMLRepository.html).
X_train, Y_train = load_dataset("eurlex-4k", "train")
X_test, Y_test = load_dataset("eurlex-4k", "test")

# Create PLT model with "eurlex-model" directory,
# it will be created and used during model training for storing weights.
# napkinXC stores already trained parts of the models to save RAM.
plt = PLT("eurlex-model")

# Fit the model on the training dataset.
# The model weights and additional data will be stored in "eurlex-model" directory.
plt.fit(X_train, Y_train)

# Predict.
Y_pred = plt.predict(X_test, top_k=1)
print("Precision at 1:", precision_at_k(Y_test, Y_pred, k=1))

# Delete plt object.
del plt

# To load the model, create a new PLT object with the same directory as the previous one.
new_plt = PLT("eurlex-model")

# Predict using a new model object.
Y_pred = new_plt.predict(X_test, top_k=1)
print("Precision at 1 after loading:", precision_at_k(Y_test, Y_pred, k=1))
