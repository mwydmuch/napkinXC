#!/usr/bin/env python3

import os
import numpy as np
from napkinxc.datasets import *
from napkinxc.metrics import *
from napkinxc.models import PLT


k = 5
precision = 4
measures = {
    #"HL": {"func": hamming_loss, "needs_weights": False},
    "P": {"func": precision_at_k, "needs_weights": False},
    "R": {"func": recall_at_k, "needs_weights": False},
    "nDCG": {"func": ndcg_at_k, "needs_weights": False},
    "PSP": {"func": psprecision_at_k, "needs_weights": True},
    #"PSnDCG": {"func": psndcg_at_k, "needs_weights": True},
    "C": {"func": coverage_at_k, "needs_weights": False},
    "mP": {"func": macro_precision_at_k, "needs_weights": False},
    "mR": {"func": macro_recall_at_k, "needs_weights": False},
    "mF1": {"func": macro_f1_measure_at_k, "needs_weights": False},
}


def evaluate(Y_true, Y_pred, max_k=5, weights=None, precision=2):
    results = {}
    for m, v in measures.items():
        r = None
        if v["needs_weights"]:
            if weights is not None:
                r = v["func"](Y_true, Y_pred, weights, k=max_k)
        else:
            r = v["func"](Y_true, Y_pred, k=max_k)
        if r is not None:
            for k in range(max_k):
                print(("{}@{}: {:." + str(precision) + "f}").format(m, k + 1, r[k] * 100))
                results[f"{m}@{k + 1}"] = r[k]
    return results


def train_model(model_file, X_train, Y_train):
    plt = PLT(model_file, verbose=False)

    if not os.path.exists(model_file):    
        plt.fit(X_train, Y_train)
    else:
        plt.load()

    return plt


dataset = "amazonCat-13K"
num_head_labels = 1000

print("Loading data ...")
X_train, Y_train = load_dataset(dataset, "train")
X_test, Y_test = load_dataset(dataset, "test")

print("Calculating inverse propensities ...")
A = 0.55
B = 1.5

if dataset in ("amazon-670k", "amazon-3M"):
    A = 0.6
    B = 2.6
elif dataset in ("wikiLSHTC", "WikipediaLarge-500K"):
    A = 0.5
    B = 0.4

inv_ps = Jain_et_al_inverse_propensity(Y_train, A=A, B=B)


# Model with all labels
os.makedirs("models", exist_ok=True)
full_model = train_model(f"models/{dataset}-model-full", X_train, Y_train)
full_pred = full_model.predict(X_test, top_k=5)

print("MODEL WITH ALL LABELS:")
full_results = evaluate(Y_test, full_pred, max_k=k, weights=inv_ps, precision=precision)
full_model.unload()

# Keep only top labels for trainingF
m = max([max(y) for y in Y_train if len(y)])
if num_head_labels < 1:
    num_head_labels = round(m * num_head_labels)

head_labels = (-count_labels(Y_train)).argsort()[:num_head_labels]
head_map = {l: i + 1 for i, l in enumerate(head_labels)}
inv_map = {i: l for l, i in head_map.items()}
                 
Y_head_train = []
for y in Y_train:
    Y_head_train.append([head_map[l] for l in y if l in head_map])

# Model with only top labels
head_model = train_model(f"models/{dataset}-model-top-{num_head_labels}", X_train, Y_head_train)
head_pred = head_model.predict(X_test, top_k=5)
head_pred_remap = []
for y in head_pred:
    head_pred_remap.append([inv_map[l] for l in y if l in inv_map])

print(f"MODEL WITH TOP {num_head_labels} LABELS:")    
head_results = evaluate(Y_test, head_pred_remap, max_k=k, weights=inv_ps, precision=precision)
head_model.unload()

print(f"DIFFERENCE BETWEEN MODELS:")
for k in full_results.keys():
    print(f"{k}: {(head_results[k] - full_results[k]) / full_results[k] * 100:.4}")
