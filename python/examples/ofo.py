from napkinxc.datasets import load_dataset
from napkinxc.models import PLT
from napkinxc.measures import f1_measure
from sklearn.model_selection import train_test_split

# Use load_dataset function to load one of the benchmark datasets
# from XML Repository (http://manikvarma.org/downloads/XC/XMLRepository.html).
X_train, Y_train = load_dataset("eurlex-4k", "train")
X_test, Y_test = load_dataset("eurlex-4k", "test")

# Using sklearn, lets split training dataset into dataset for training and tuning thresholds for macro F-measure.
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

# Create Probabilistic Labels Tree model and fit it on the training dataset.
# The model weights and additional data will be stored in "eurlex-model" directory.
plt = PLT("eurlex-model")
plt.fit(X_train, Y_train)

# First lets check macro F1 measure performance with const. threshold = 0.5
Y_pred_single_th = plt.predict(X_test, threshold=0.5)
print("Micro F1 measure with const. threshold = 0.5:", f1_measure(Y_test, Y_pred_single_th, average='micro', zero_division=0))
print("Macro F1 measure with const. threshold = 0.5:", f1_measure(Y_test, Y_pred_single_th, average='macro', zero_division=0))

# Now lets use Online F measure optimization procedure to find better thresholds.
# OFO can be used to find optimal threshold/thresholds for micro F1 measure and macro F1 measure.
micro_ths = plt.ofo(X_valid, Y_valid, type="micro", a=1, b=2, epochs=5)
macro_ths = plt.ofo(X_valid, Y_valid, type="macro", a=1, b=2, epochs=10)

# Lets predict with the new thresholds and compare the results
Y_pred_micro_ths = plt.predict(X_test, threshold=micro_ths)
Y_pred_macro_ths = plt.predict(X_test, threshold=macro_ths)

print("Micro F1 measure with thresholds from OFO procedure:", f1_measure(Y_test, Y_pred_micro_ths, average='micro', zero_division=0))
print("Macro F1 measure with thresholds from OFO procedure:", f1_measure(Y_test, Y_pred_macro_ths, average='macro', zero_division=0))
