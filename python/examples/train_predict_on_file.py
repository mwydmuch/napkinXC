#!/usr/bin/env python3

# This demo shows the how to try the model on data stored in text file.
# During training model, napkinXC needs to load data to its internal format.
# This double memory consumption and might be a problem for large dataset.
# Instead of loading data to memory, one can learn model on 

from napkinxc.datasets import download_dataset, load_dataset, save_libsvm_file
from napkinxc.models import PLT
from napkinxc.metrics import precision_at_k

# Use download_dataset to download data
# from XML Repository (http://manikvarma.org/downloads/XC/XMLRepository.html).
print("Downloading data ...")
download_dataset("eurlex-4k")

# Create Probabilistic Labels Tree model,
# directory "eurlex-model" will be created and used during model training.
# napkinXC stores already trained parts of the model to save RAM.
# Model directory is only a required argument for model constructors.
plt = PLT("eurlex-model")

# Fit the model on the training dataset stored in data/Eurlex/eurlex_train.txt.
# The model weights and additional data will be stored in "eurlex-model" directory.
print("Fitting model...")
plt.fit_on_file("data/Eurlex/eurlex_train.txt")

# After the training model is not loaded to RAM.
# You can preload the model to RAM to perform prediction.
plt.load()

# Predict only five top labels for each data point in the test dataset.
# This will also load the model if it is not loaded.
print("Predicting ...")
Y_pred = plt.predict_for_file("data/Eurlex/eurlex_test.txt", top_k=5)

# Evaluate the prediction with precision at 5 measure.
_, Y_test = load_dataset("eurlex-4k", "test")
print("Precision at k:", precision_at_k(Y_test, Y_pred, k=5))

# Unload the model from RAM
# You can also just delete the object if you do not need it
plt.unload()
