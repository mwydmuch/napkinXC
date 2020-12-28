from napkinxc.datasets import load_dataset
from napkinxc.models import PLT
from napkinxc.measures import precision_at_k
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.utils.estimator_checks import check_estimator

# Use load_dataset function to load one of the benchmark datasets
# from XML Repository (http://manikvarma.org/downloads/XC/XMLRepository.html).
X_train, Y_train = load_dataset("eurlex-4k", "train")
X_test, Y_test = load_dataset("eurlex-4k", "test")

# Using sklearn, lets split training dataset into dataset for training and tuning thresholds for macro F-measure.
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=20, random_state=0)

# Create Probabilistic Labels Tree model and fit it on the training dataset.
# The model weights and additional data will be stored in "eurlex-model" directory.
plt = PLT("eurlex-model")
plt.fit(X_train, Y_train)

# This will also load the model if it is not loaded.
ths = plt.ofo(X_valid, Y_valid, type="micro", a=10, b=20, epochs=5)

# Lets compare prediction with thresholds with prediction with const threshold = 0.5
Y_pred_ths = plt.predict(X_test, threshold=ths)
Y_pred_single_th = plt.predict(X_test, threshold=0.5)
