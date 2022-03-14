#!/usr/bin/env python

import sys
import os
import shutil

napkinxc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../python")
sys.path.append(napkinxc_path)

from napkinxc.datasets import download_dataset, _get_data_meta

aliases = { 
    "yeast": "yeast",
    "mediamill": "mediamill",
    "rcv1x": "RCV1X-2K",
    "amazonCat": "AmazonCat-13K",
    "amazonCat-14K": "AmazonCat-14K",
    "amazon": "Amazon-670K",
    "amazon-3M": "Amazon-3M",
    "deliciousLarge": "Delicious-200K",
    "eurlex": "EURLex-4K",
    "wiki10": "Wiki10-31K",
    "wikiLSHTC": "WikiLSHTC-325K",
    "WikipediaLarge-500K": "Wikipedia-500K",
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Requires true and prediction files as arguments!")
        exit(1)

    dataset = aliases.get(sys.argv[1], sys.argv[1])

    root = "data"
    if len(sys.argv) >= 3:
        root = sys.argv[2]

    dataset_meta = _get_data_meta(dataset, format='bow-v1')
    download_dataset(dataset, format='bow-v1', root=root, verbose=True)
    shutil.move(os.path.join(root, dataset_meta['dir']), os.path.join(root, sys.argv[1]))
