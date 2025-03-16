#!/usr/bin/env python3

# This demo shows the most basic usage of the napkinXC library.

from napkinxc.datasets import load_dataset
from napkinxc.models import PLT
from napkinxc.metrics import precision_at_k

# Use load_dataset function to load one of the benchmark datasets
# from XML Repository (http://manikvarma.org/downloads/XC/XMLRepository.html).
print("Loading data ...")
X_train, Y_train = load_dataset("eurlex-4k", "train")
X_test, Y_test = load_dataset("eurlex-4k", "test")

# Create Probabilistic Labels Tree model,
# directory "eurlex-model" will be created and used during model training.
# napkinXC stores already trained parts of the model to save RAM.
# Model directory is only a required argument for model constructors.
plt = PLT("eurlex-model")

# Fit the model on the training dataset.
# The model weights and additional data will be stored in "eurlex-model" directory.
# Features matrix X should be SciPy csr_matrix, NumPy array, or list of tuples of (idx, value),
# while labels matrix Y should be list of lists or tuples containing positive labels, SciPy csr_matrix, NumPy array.
print("Fitting model...")
plt.fit(X_train, Y_train)

# After the training model is not loaded to RAM.
# You can preload the model to RAM to perform prediction.
plt.load()

# Predict only five top labels for each data point in the test dataset.
# This will also load the model if it is not loaded.
print("Predicting ...")
Y_pred = plt.predict(X_test, top_k=5)
print("Predictions for the first example:", Y_pred[0])

# To predict labels with probabilty estimates use predict_proba instead.
print("Predicting with probabilities ...")
Y_pred_proba = plt.predict_proba(X_test, top_k=5)
print("Predictions for the first example with probabilities:", Y_pred_proba[0])

# Evaluate the prediction with precision at 5 measure (both types of predictions can be used).
print("Precision at k:", precision_at_k(Y_test, Y_pred, k=5))
# or print("Precision at k:", precision_at_k(Y_test, Y_pred_proba, k=5))

# Unload the model from RAM
# You can also just delete the object if you do not need it
plt.unload()
