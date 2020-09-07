#!/usr/bin/env python3

from napkinxc.datasets import load_dataset
from napkinxc.models import PLT
from napkinxc.measures import precision_at_k

# Use load_dataset function to load one of the benchmark datasets
# from XML Repository (http://manikvarma.org/downloads/XC/XMLRepository.html)
X_train, Y_train = load_dataset("eurlex-4k", "train")
X_test, Y_test = load_dataset("eurlex-4k", "test")

# Create Probabilistic Labels Tree models,
# directory "eurlex-model" will be created and used for model training and storing
plt = PLT("eurlex-model")

# Fit the model on the train dataset
plt.fit(X_train, Y_train)

# Predict only the best label for each datapoint in the test dataset
Y_pred = plt.predict(X_test, top_k=1)

# Evaluate precision at 1
print(precision_at_k(Y_test, Y_pred, k=1))
