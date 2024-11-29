#!/usr/bin/env python

import sys
import os
import shutil

file_dir = os.path.dirname(os.path.abspath(__file__))
napkinxc_path = os.path.join(file_dir, "../python")
sys.path.extend([file_dir, napkinxc_path])

from napkinxc.datasets import download_dataset, _get_data_meta

old_aliases = {
    "rcv1x": "RCV1-2K",
    "amazonCat": "AmazonCat-13K",
    "amazonCat-14K": "AmazonCat-14K",
    "amazon": "Amazon-670K",
    "amazon-3M": "Amazon-3M",
    "deliciousLarge": "DeliciousLarge-200K",
    "eurlex": "EURLex-4K",
    "wiki10": "Wiki10-31K",
    "wikiLSHTC": "WikiLSHTC-325K",
    "WikipediaLarge-500K": "WikipediaLarge-500K",
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: download_dataset.py [dataset name] [format (optional, defaults to \"bow\")] [root dir (optional, defaults to \"data\")]")
        exit(1)

    dataset = old_aliases.get(sys.argv[1], sys.argv[1])

    format = "bow"
    if len(sys.argv) >= 3:
        format = sys.argv[2]

    root = "data"
    if len(sys.argv) >= 4:
        root = sys.argv[3]

    dataset_meta = _get_data_meta(dataset, format=format)
    download_dataset(dataset, format=format, root=root, verbose=True)
    shutil.move(os.path.join(root, dataset_meta['dir']), os.path.join(root, sys.argv[1]))
