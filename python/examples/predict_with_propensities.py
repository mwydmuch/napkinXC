#!/usr/bin/env python3

# This demo shows the most basic usage of the napkinXC library.

from napkinxc.datasets import load_dataset
from napkinxc.models import PLT
from napkinxc.measures import precision_at_k, psprecision_at_k, inverse_propensity

# Use load_dataset function to load one of the benchmark datasets
# from XML Repository (http://manikvarma.org/downloads/XC/XMLRepository.html).
X_train, Y_train = load_dataset("eurlex-4k", "train")
X_test, Y_test = load_dataset("eurlex-4k", "test")

# Create Probabilistic Labels Tree model,
# directory "eurlex-model" will be created and used during model training.
# napkinXC stores already trained parts of the model to save RAM.
# Model directory is only a required argument for model constructors.
plt = PLT("eurlex-model")

# Fit the model on the training (observed) dataset.
# The model weights and additional data will be stored in "eurlex-model" directory.
# Features matrix X must be SciPy csr_matrix, NumPy array, or list of tuples of (idx, value),
# while labels matrix Y should be list of lists or tuples containing positive labels.
plt.fit(X_train, Y_train)

# After the training model is not loaded to RAM.
# You can preload the model to RAM to perform prediction.
plt.load()

# Predict five top labels for each data point in the test dataset using standard uniform-cost search
Y_pred = plt.predict(X_test, top_k=5)

# Calculate inverse propensity values (aka propensity scores) and predict with label weights
inv_ps = inverse_propensity(Y_train, A=0.55, B=1.5)
ps_Y_pred = plt.predict(X_test, labels_weights=inv_ps, top_k=5)

# Evaluate the both predictions with propensity-scored and vanilla precision at 5 measure.
print("Standard prediction:")
print("  Precision at k:", precision_at_k(Y_test, Y_pred, k=5))
print("  Propensity-scored precision at k:", psprecision_at_k(Y_test, Y_pred, inv_ps, k=5))

print("Prediction weighted by inverse propensity:")
print("  Precision at k:", precision_at_k(Y_test, ps_Y_pred, k=5))
print("  Propensity-scored precision at k:", psprecision_at_k(Y_test, ps_Y_pred, inv_ps, k=5))
