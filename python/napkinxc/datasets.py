# Copyright (c) 2020-2022 by Marek Wydmuch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gdown
import gzip
import json
import re
import requests
import warnings
import zipfile
from sys import stdout
from os import makedirs, path, remove
import numpy as np
from scipy.sparse import csr_matrix

try:
    from ._napkinxc import _load_libsvm_file_labels_list, _load_libsvm_file_labels_csr_matrix
except ImportError:
    warnings.warn("Couldn't import napkinXC cpp module, some functions may fail.")


# List of all available datasets
DATASETS = {
    # Small datasets, for testing
    'yeast': {
        'name': 'yeast',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1qcU-jxF7VmHBvwTn0-6MB4FW4ItD-bIp',  # Marek Wydmuch's upload
            'dir': 'yeast',
            'train': 'yeast_train.txt',
            'test': 'yeast_test.txt',
            'file_format': 'libsvm',
        }
    },
    'mediamill': {
        'name': 'mediamill',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1YDlU9GTYSoydU1UYTNT8BpPkRw1BCnFw',  # Marek Wydmuch's upload
            'dir': 'mediamill',
            'train': 'mediamill_train.txt',
            'test': 'mediamill_test.txt',
            'file_format': 'libsvm',
        }
    },

    # XMLC repo datasets
    'rcv1-2k': {
        'name': 'RCV1-2K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1pdken_Di3PLcLlRbhHF4XIWp-CEiv9oQ',  # Marek Wydmuch's upload
            'dir': 'RCV1-x',
            'train': 'rcv1x_train.txt',
            'test': 'rcv1x_test.txt',
            'file_format': 'libsvm',
        }
    },
    'eurlex-4k': {
        'name': 'EURLex-4K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            # 'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGU0VTR1pCejFpWjg',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1xWW1UykQBTD2IorVn7V8XubJatQDScol',  # Marek Wydmuch's upload
            'dir': 'Eurlex',
            'train': 'eurlex_train.txt',
            'test': 'eurlex_test.txt',
            'file_format': 'libsvm',
        }
    },
    'eurlex-4.3k': {
        'name': 'EURLex-4.3K',
        'formats': ['bow'],
        'subsets': ['train', 'test', 'validation'],
        'bow': {
            # 'url': 'https://drive.google.com/uc?export=download&id=1b3mWgaKIAmc9Ae3E0QrokiIFA9Qj1K9r',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1UZ5joZdU1axWdCVSECakmEA8XQJ2gVuQ',  # Marek Wydmuch's upload
            'dir': 'EURLex-4.3K',
            'train': 'train.txt',
            'test': 'test.txt',
            'validation': 'validation.txt',
            'file_format': 'libsvm',
        }
    },
    'amazoncat-13k': {
        'name': 'AmazonCat-13K',
        'formats': ['bow-v1', 'bow', 'raw'],
        'subsets': ['train', 'test'],
        'bow-v1': {
            # 'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGa2tMbVJGdDNSMGc',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1WvBWF9hH5ZlFcpCRhp8wLVGwdeGZXKD2',  # Marek Wydmuch's upload
            'dir': 'AmazonCat',
            'train': 'amazonCat_train.txt',
            'test': 'amazonCat_test.txt',
            'file_format': 'libsvm',
        },
        'bow': {  # v2
            'url': 'https://drive.google.com/uc?export=download&id=17EFQtnRswEv0XyPng2EOy5IeWmeZbe0a',  # Marek Wydmuch's upload
            'dir': 'AmazonCat-13K.bow',
            'train': 'train.txt',
            'test': 'test.txt',
            'file_format': 'libsvm',
        },
        'raw': {
            # 'url': 'https://drive.google.com/uc?export=download&id=17rVRDarPwlMpb3l5zof9h34FlwbpTu4l',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1iOsEOYL-fY-GOt0CNfgW9YO2oFDTb6X4',  # Marek Wydmuch's upload
            'dir': 'AmazonCat-13K.raw',
            'train': 'trn.json.gz',
            'test': 'tst.json.gz',
            'file_format': 'jsonlines',
            'features_fields': ['title', 'content'],
            'labels_field': 'target_ind'
        }
    },
    'amazoncat-14k': {
        'name': 'AmazonCat-14K',
        'formats': ['bow-v1', 'bow', 'raw'],
        'subsets': ['train', 'test'],
        'bow-v1': {
            # 'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGaDFqU2E5U0dxS00',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=19IhTX1_a4U5I2Q56VUkzq2jFBpwm2-n-',  # Marek Wydmuch's upload
            'dir': 'AmazonCat-14K',
            'train': 'amazonCat-14K_train.txt',
            'test': 'amazonCat-14K_test.txt',
            'file_format': 'libsvm',
        },
        'bow': {  # v2
            'url': 'https://drive.google.com/uc?export=download&id=1VAxPvByB-SX3_MefdfW1eKZYTfz0vMcR',  # Marek Wydmuch's upload
            'dir': 'AmazonCat-14K.bow',
            'train': 'train.txt',
            'test': 'test.txt',
            'file_format': 'libsvm',
        },
        'raw': {
            # 'url': 'https://drive.google.com/uc?export=download&id=1vy1N-lDdDfuoo0CNwFE11hb3INCpJHFx',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1CTu0ZKhpaWuzIoOheDrccG_skXhyfzLm',  # Marek Wydmuch's upload
            'dir': 'AmazonCat-14K.raw',
            'train': 'trn.json.gz',
            'test': 'tst.json.gz',
            'file_format': 'jsonlines',
            'features_fields': ['title', 'content'],
            'labels_field': 'target_ind'
        }
    },
    'wiki10-31k': {
        'name': 'Wiki10-31K',
        'formats': ['bow-v1', 'bow'],
        'subsets': ['train', 'test'],
        'bow-v1': {
            # 'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGaDdOeGliWF9EOTA',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1uV-p682ggXQQTiKZyK5M6B0xJJk8_lJx',  # Marek Wydmuch's upload
            'dir': 'Wiki10',
            'train': 'wiki10_train.txt',
            'test': 'wiki10_test.txt',
            'file_format': 'libsvm',
        },
        'bow': {  # v2
            'url': 'https://drive.google.com/uc?export=download&id=1WbxhKvvBSynX5_Sf6OaPReTMs8K6bZ8u',  # Marek Wydmuch's upload
            'dir': 'Wiki10.bow',
            'train': 'train.txt',
            'test': 'test.txt',
            'file_format': 'libsvm',
        }
    },
    'deliciouslarge-200k': {
        'name': 'DeliciousLarge-200K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            # 'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGR3lBWWYyVlhDLWM',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=18Tb595TuGyFW--nEYGfduwS5WvSbX-sX',  # Marek Wydmuch's upload
            'dir': 'DeliciousLarge',
            'train': 'deliciousLarge_train.txt',
            'test': 'deliciousLarge_test.txt',
            'file_format': 'libsvm',
        }
    },
    'wikilshtc-325k': {
        'name': 'WikiLSHTC-325K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            # 'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGSHE1SWx4TVRva3c',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1Y5xEJu-j3M7voyFV5Ouq9Wl3vyd9fus2',  # Marek Wydmuch's upload
            'dir': 'WikiLSHTC',
            'train': 'wikiLSHTC_train.txt',
            'test': 'wikiLSHTC_test.txt',
            'file_format': 'libsvm',
        }
    },
    'wikiseealsotitles-350k': {
        'name': 'WikiSeeAlsoTitles-350K',
        'formats': ['bow', 'raw'],
        'subsets': ['train', 'test'],
        'bow': {
            # 'url': 'https://drive.google.com/uc?export=download&id=1bHtiLVF5EFsVL3qyU7y5e3M-fYHvXsG9',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1icZIEU2TA5k86mTweciSa3mBLFD3fbIP',  # Marek Wydmuch's upload
            # 'dir': 'WikiSeeAlsoTitles-350K',  # (XMLC version)
            'dir': 'WikiSeeAlsoTitles-350K.bow',
            'train': {'X': 'trn_X_Xf.txt', 'Y': 'trn_X_Y.txt'},
            'test': {'X': 'tst_X_Xf.txt', 'Y': 'tst_X_Y.txt'},
            'file_format': 'XY_sparse',
        },
        'raw': {
            # 'url': 'https://drive.google.com/uc?export=download&id=1sxPHzlnotUKjbtVRe0GuXGfSI7YhBSdc',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=13Y4cvJ6zKDUpIdBVsaBHp1rEWiv5tyQy',  # Marek Wydmuch's upload
            # 'dir': 'WikiSeeAlsoTItles-350K',  # (XMLC version)
            'dir': 'WikiSeeAlsoTitles-350K.raw',
            'train': 'trn.json.gz', # Type is there on purpose
            'test': 'tst.json.gz',
            'file_format': 'jsonlines',
            'features_fields': ['title', 'content'],
            'labels_field': 'target_ind'
        }
    },
    'wikititles-500k': {
        'name': 'WikiTitles-500K',
        'formats': ['bow', 'raw'],
        'subsets': ['train', 'test'],
        'bow': {
            # 'url': 'https://drive.google.com/uc?export=download&id=11U4qDWKvsR6pCzLvY3APckx-R_ihyMih',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1RKKOw0yl2RxrWYbb-S508aOwKdCjY_ih',  # Marek Wydmuch's upload
            'dir': 'WikiTitles-500K',
            'train': {'X': 'trn_X_Xf.txt', 'Y': 'trn_X_Y.txt'},
            'test': {'X': 'tst_X_Xf.txt', 'Y': 'tst_X_Y.txt'},
            'file_format': 'XY_sparse',
        },
        'raw': {
            'url': 'https://drive.google.com/uc?export=download&id=1YStqoa6_5Qxd9FpExTNt-_tcXUhYXUFM',  # Marek Wydmuch's reupload
            'dir': 'Wikipedia-500K.raw',
            'train': 'trn.raw.json.gz',
            'test': 'tst.raw.json.gz',
            'file_format': 'jsonlines',
            'features_fields': ['title'],
            'labels_field': 'target_ind'
        }
    },
    'wikipedialarge-500k': {
        'name': 'WikipediaLarge-500K',
        'formats': ['bow', 'raw'],
        'subsets': ['train', 'test'],
        'bow': {
            # 'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGRmEzVDVkNjBMR3c',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1LkGdINF5coOfkm6J2xys3V1NMW7zz8U1',  # Marek Wydmuch's upload
            'dir': 'WikipediaLarge-500K',
            'train': 'WikipediaLarge-500K_train.txt',
            'test': 'WikipediaLarge-500K_test.txt',
            'file_format': 'libsvm',
        },
        'raw': {
            'url': 'https://drive.google.com/uc?export=download&id=1YStqoa6_5Qxd9FpExTNt-_tcXUhYXUFM',  # Marek Wydmuch's reupload
            'dir': 'Wikipedia-500K.raw',
            'train': 'trn.raw.json.gz',
            'test': 'tst.raw.json.gz',
            'file_format': 'jsonlines',
            'features_fields': ['title', 'content'],
            'labels_field': 'target_ind'
        }
    },
    'amazontitles-670k': {
        'name': 'AmazonTitles-670K',
        'formats': ['bow', 'raw'],
        'subsets': ['train', 'test'],
        'bow': {
            # 'url': 'https://drive.google.com/uc?export=download&id=1OKnaLu4SDMOQ69rHdwF8ExKkeF2SZw7z',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1TWF2_tPGdezuljGEHbp63pe7eQUq7wK6',  # Marek Wydmuch's upload
            'dir': 'AmazonTitles-670K',
            'train': {'X': 'trn_X_Xf.txt', 'Y': 'trn_X_Y.txt'},
            'test': {'X': 'tst_X_Xf.txt', 'Y': 'tst_X_Y.txt'},
            'file_format': 'XY_sparse',
        },
        'raw': {
            # 'url': 'https://drive.google.com/uc?export=download&id=1FPqD8Wns7NXTSYDAcK4ZsqUUGABLyZMn',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1f1w0cGc-jJoPxJRY7v7BvKJuelspwOZV',  # Marek Wydmuch's upload
            'dir': 'AmazonTitles-670K',
            'train': 'trn.json.gz',
            'test': 'tst.json.gz',
            'file_format': 'jsonlines',
            'features_fields': ['title'],
            'labels_field': 'target_ind'
        }
    },
    'amazon-670k': {
        'name': 'Amazon-670K',
        'formats': ['bow-v1', 'bow', 'raw'],
        'subsets': ['train', 'test'],
        'bow-v1': {
            # 'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGdUJwRzltS1dvUVk',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1PV-wbKVv6Ng1K1XM1USqNxFhaQo2qfk6',  # Marek Wydmuch's upload
            'dir': 'Amazon',
            'train': 'amazon_train.txt',
            'test': 'amazon_test.txt',
            'file_format': 'libsvm',
        },
        'bow': {
            # 'url': 'https://drive.google.com/uc?export=download&id=1TLaXCNB_IDtLhk4ycOnyud0PswWAW6hR',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1d5EMgVxjjDBGjw7M09gaygx5wSt4-AqQ',  # Marek Wydmuch's upload
            # 'dir': 'Amazon670K.bow',  # (XMLC version)
            'dir': 'Amazon-670K.bow',
            'train': 'train.txt',
            'test': 'test.txt',
            'file_format': 'libsvm',
        },
        'raw': {
            # 'url': 'https://drive.google.com/uc?export=download&id=16FIzX3TnlsqbrwSJJ2gDih69laezfZWR',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1WAzs7qfLY8ZoluHbY55WyakhibW27kS2',  # Marek Wydmuch's upload
            'dir': 'Amazon-670K.raw',
            'train': 'trn.raw.json.gz',
            'test': 'tst.raw.json.gz',
            'file_format': 'jsonlines',
            'features_fields': ['title', 'content'],
            'labels_field': 'target_ind'
        }
    },
    'amazontitles-3m': {
        'name': 'AmazonTitles-3M',
        'formats': ['bow', 'raw'],
        'subsets': ['train', 'test'],
        'bow': {
            # 'url': 'https://drive.google.com/uc?export=download&id=1PGzippnzIcgVNYZ8qQKVb0GNiARjQxvV',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=15MpbBO7MTRLc1okzwSiPkQ9ZHgyKMzzm',  # Marek Wydmuch's upload
            'dir': 'AmazonTitles-3M',
            'train': {'X': 'trn_X_Xf.txt', 'Y': 'trn_X_Y.txt'},
            'test': {'X': 'tst_X_Xf.txt', 'Y': 'tst_X_Y.txt'},
            'file_format': 'XY_sparse',
        },
        'raw': {
            # 'url': 'https://drive.google.com/uc?export=download&id=1m0MMApC0vPpjEfI35SAaBGqKDsYypVXs',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=18Rpy2WWJL8AlGoipT3DSWehzu54QLH_7',  # Marek Wydmuch's upload
            'dir': 'AmazonTitles-3M',
            'train': 'trn.json.gz',
            'test': 'tst.json.gz',
            'file_format': 'jsonlines',
            'features_fields': ['title'],
            'labels_field': 'target_ind'
        }
    },
    'amazon-3m': {
        'name': 'Amazon-3M',
        'formats': ['bow-v1', 'bow', 'raw'],
        'subsets': ['train', 'test'],
        'bow-v1': {
            # 'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGUEd4eTRxaWl3YkE',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1ork7yeAcliD9JQiRdEx4QknhMFHjVrku',  # Marek Wydmuch's upload
            'dir': 'Amazon-3M',
            'train': 'amazon-3M_train.txt',
            'test': 'amazon-3M_test.txt',
            'file_format': 'libsvm',
        },
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1tepPa5oOSp7jmih7LTEemHOL9KwdBtuB',  # Marek Wydmuch's upload
            'dir': 'Amazon-3M.bow',
            'train': 'train.txt',
            'test': 'test.txt',
            'file_format': 'libsvm',
        },
        'raw': {
            # 'url': 'https://drive.google.com/uc?export=download&id=1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN',  # XMLC repo url (stopped working)
            'url': 'https://drive.google.com/uc?export=download&id=1oDW_-c9dW6Dwj0ja_RVoLRNp87JKw1gh',  # Marek Wydmuch's upload
            'dir': 'Amazon-3M.raw',
            'train': 'trn.json.gz',
            'test': 'tst.json.gz',
            'file_format': 'jsonlines',
            'features_fields': ['title', 'content'],
            'labels_field': 'target_ind'
        }
    },
    'lf-amazontitles-131k': {
        'name': 'LF-AmazonTitles-131K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1VlfcdJKJA99223fLEawRmrXhXpwjwJKn',  # XMLC repo url
            'dir': 'LF-AmazonTitles-131K',
            'train': 'train.txt',
            'test': 'test.txt',
            'file_format': 'libsvm',
        }
    },
    'lf-amazon-131k': {
        'name': 'LF-Amazon-131K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1YNGEifTHu4qWBmCaLEBfjx07qRqw9DVW',  # XMLC repo url
            'dir': 'LF-Amazon-131K',
            'train': 'train.txt',
            'test': 'test.txt',
            'file_format': 'libsvm',
        }
    },
    'lf-wikiseealsotitles-320k': {
        'name': 'LF-WikiSeeAlsoTitles-320K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1edWtizAFBbUzxo9Z2wipGSEA9bfy5mdX',  # XMLC repo url
            'dir': 'LF-WikiSeeAlsoTitles-320K',
            'train': 'train.txt',
            'test': 'test.txt',
            'file_format': 'libsvm',
        }
    },
    'lf-wikiseealso-320k': {
        'name': 'LF-WikiSeeAlso-320K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1N8C_RL71ErX6X92ew9h8qRuTWJ9LywE8',  # XMLC repo url
            'dir': 'LF-WikiSeeAlso-320K',
            'train': 'train.txt',
            'test': 'test.txt',
            'file_format': 'libsvm',
        }
    },
    'lf-wikititles-500k': {
        'name': 'LF-WikiTitles-500K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1qa1HTTD509J5r4yNAH-Aq6wArljgg4mx',  # XMLC repo url
            'dir': 'LF-WikiTitles-500K',
            'train': 'train.txt',
            'test': 'test.txt',
            'file_format': 'libsvm',
        }
    },
    'lf-amazontitles-1.3m': {
        'name': 'LF-AmazonTitles-1.3M',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1Davc6BIfoTIAS3mP1mUY5EGcGr2zN2pO',  # XMLC repo url
            'dir': 'LF-AmazonTitles-1.3M',
            'train': 'train.txt',
            'test': 'test.txt',
            'file_format': 'libsvm',
        }
    },
}


# Main functions for downloading and loading datasets
def load_libsvm_file(file, labels_format="list", sort_indices=False):
    """
    Load data in the libsvm format into sparse CSR matrix.
    The format is text-based. Each line contains an instance and is ended by a ``\\n`` character.

    .. code::

        <label>,<label>,... <feature>(:<value>) <feature>(:<value>) ...

    ``<label>`` and ``<feature>`` are indexes that should be positive integers.
    This method supports less-strict versions of the format.
    Labels and features do not have to be sorted in ascending order.
    The ``:<value>`` can be omitted after ``<feature>``, to assume value = 1.
    It automatically detects header used in format of datasets from
    `The Extreme Classification Repository <https://manikvarma.github.io/downloads/XC/XMLRepository.html>`_,

    :param file: Path to a file to load
    :type file: str
    :param labels_format: Format in which load the labels data (``'list'`` or ``'csr_matrix'``), defaults to `csr_matrix`
    :type labels_format: str
    :param sort_indices: If True, sort indices, otherwise keep original order, defaults to True
    :type sort_indices: bool
    :return:  Features and labels data
    :rtype: (csr_matrix, list[list[int]]) or (csr_matrix, csr_matrix)
    """
    if labels_format == 'list':
        labels, features = _load_libsvm_file_labels_list(file, sort_indices)
        return csr_matrix(features), labels
    elif labels_format == 'csr_matrix':
        labels, features = _load_libsvm_file_labels_csr_matrix(file, sort_indices)
        return csr_matrix(features), csr_matrix(labels)
    else:
        raise ValueError("Label format {} is not valid format".format(labels_format))


def load_json_lines_file(file, features_fields=['title', 'content'], labels_field='target_ind', gzip_file=None):
    """
    Load data in the JSON lines format into list of features and list of labels.
    
    :param file: Path to a JSON lines file to load
    :type file: str
    :param features_fields: list of fields of JSON line that contain features, fields will be concatenated in the specified order, defaults to ['title', 'content']
    :type features_fields: list[str], optional
    :param labels_field: field name that contains labels, defaults to ``'target_ind'``
    :type labels_field: str, optional
    :param gzip_file: If True, read file as gzip file, if None, decide based on file extension, defaults to None
    :type gzip_file: bool, optional
    :return: Raw text of documents and labels
    :rtype: (list[str], list[list[int|str]])
    """

    X = []
    Y = []
    if gzip_file == True or file[-3:] == '.gz':
        f = gzip.open(file, 'rb')
    else:
        f = open(file, 'r')
    for line in f:
        data = json.loads(line)
        if not all(f in data for f in features_fields):
            raise ValueError("Not all features fields {} are not in {}".format(features_fields, data))
        if not labels_field in data:
            raise ValueError("Labels field {} is not not in {}".format(labels_field, data))
        X.append(' '.join([data[f] for f in features_fields]))
        Y.append(data[labels_field])

    return X, Y


def download_dataset(dataset, subset='train', format='bow', root='./data', verbose=False):
    """
    Downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again.

    :param dataset: Name of the dataset to load, case insensitive, available datasets:

        - ``'Eurlex-4K'`` (``'bow'`` format only),
        - ``'Eurlex-4.3K'`` (``'bow'`` format only),
        - ``'AmazonCat-13K'``,
        - ``'AmazonCat-14K'``,
        - ``'Wiki10-31K'`` (alias: ``'Wiki10'``, ``'bow'`` format only),
        - ``'DeliciousLarge-200K'`` (alias: ``'DeliciousLarge'``, ``'bow'`` format only)
        - ``'WikiLSHTC-325K'`` (alias: ``'WikiLSHTC'``, ``'bow'`` format only)
        - ``'WikiSeeAlsoTitles-350K'``,
        - ``'WikiTitles-500K'``,
        - ``'WikipediaLarge-500K'`` (alias: ``'WikipediaLarge'``),
        - ``'AmazonTitles-670K'``,
        - ``'Amazon-670K'``,
        - ``'AmazonTitles-3M'``,
        - ``'Amazon-3M'``,
        - ``'LF-AmazonTitles-131K'`` (for now ``'bow'`` format only),
        - ``'LF-Amazon-131K'`` (for now ``'bow'`` format only),
        - ``'LF-WikiSeeAlsoTitles-320K'`` (for now ``'bow'`` format only),
        - ``'LF-WikiSeeAlso-320K'`` (for now ``'bow'`` format only),
        - ``'LF-WikiTitles-500K'`` (for now ``'bow'`` format only),
        - ``'LF-AmazonTitles-1.3M'`` (for now ``'bow'`` format only).

    :type dataset: str
    :param subset: Subset of dataset to download {``'train'``, ``'test'``, ``'validation'``}, defaults to ``'train'``
    :type subset: str, optional
    :param format: Format of dataset to load {``'bow'`` (bag-of-words/tf-idf weights, alias ``'tf-idf'``), ``'raw'`` (raw text)}, defaults to ``'bow'``
    :type format: str, optional
    :param root: Location of datasets directory, defaults to ``'./data'``
    :type root: str, optional
    :param verbose: If True print downloading and loading progress, defaults to False
    :type verbose: bool, optional
    """
    dataset_meta = _get_data_meta(dataset, subset=subset, format=format)
    dataset_dest = path.join(root, dataset.lower() + '_' + format + ".zip")
    data_dir = path.join(root, dataset_meta['dir'])
    file_path = dataset_meta[subset]

    if isinstance(file_path, str):
        file_path = [file_path]
    elif isinstance(file_path, dict):
        file_path = file_path.values()
    if not all(path.exists(path.join(data_dir, f)) for f in file_path):
        if 'drive.google.com' in dataset_meta['url']:
            _download_file_from_google_drive(dataset_meta['url'], dataset_dest, unzip=True, overwrite=True, delete_zip=True, verbose=verbose)


def load_dataset(dataset, subset='train', format='bow', root='./data', verbose=False):
    """
    Downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again.
    Then loads requested datasets into features matrix and labels.

    :param dataset: Name of the dataset to load, case insensitive, available datasets:

        - ``'Eurlex-4K'`` (``'bow'`` format only),
        - ``'Eurlex-4.3K'`` (``'bow'`` format only),
        - ``'AmazonCat-13K'``,
        - ``'AmazonCat-14K'``,
        - ``'Wiki10-31K'`` (alias: ``'Wiki10'``, ``'bow'`` format only),
        - ``'DeliciousLarge-200K'`` (alias: ``'DeliciousLarge'``, ``'bow'`` format only)
        - ``'WikiLSHTC-325K'`` (alias: ``'WikiLSHTC'``, ``'bow'`` format only)
        - ``'WikiSeeAlsoTitles-350K'``,
        - ``'WikiTitles-500K'``,
        - ``'WikipediaLarge-500K'`` (alias: ``'WikipediaLarge'``),
        - ``'AmazonTitles-670K'``,
        - ``'Amazon-670K'``,
        - ``'AmazonTitles-3M'``,
        - ``'Amazon-3M'``,
        - ``'LF-AmazonTitles-131K'`` (for now ``'bow'`` format only),
        - ``'LF-Amazon-131K'`` (for now ``'bow'`` format only),
        - ``'LF-WikiSeeAlsoTitles-320K'`` (for now ``'bow'`` format only),
        - ``'LF-WikiSeeAlso-320K'`` (for now ``'bow'`` format only),
        - ``'LF-WikiTitles-500K'`` (for now ``'bow'`` format only),
        - ``'LF-AmazonTitles-1.3M'`` (for now ``'bow'`` format only).

    :type dataset: str
    :param subset: Subset of dataset to load into features matrix and labels {``'train'``, ``'test'``, ``'validation'``}, defaults to ``'train'``
    :type subset: str, optional
    :param format: Format of dataset to load {``'bow'`` (bag-of-words/tf-idf weights, alias ``'tf-idf'``), ``'raw'`` (raw text)}, defaults to ``'bow'``
    :type format: str, optional
    :param root: Location of datasets directory, defaults to ``'./data'``
    :type root: str, optional
    :param verbose: If True print downloading and loading progress, defaults to False
    :type verbose: bool, optional
    :return: Tuple of features matrix and labels.
    :rtype: (csr_matrix, list[list[int]]) or (list[str], list[list[str]])
    """
    download_dataset(dataset, subset=subset, format=format, root=root, verbose=verbose)
    dataset_meta = _get_data_meta(dataset, subset=subset, format=format)
    file_format = dataset_meta['file_format']
    data_dir = path.join(root, dataset_meta['dir'])
    file_path = dataset_meta[subset]

    if file_format == 'libsvm':
        return load_libsvm_file(path.join(data_dir, file_path))
    elif file_format == 'XY_sparse':
        X, _ = load_libsvm_file(path.join(data_dir, file_path['X']))
        Y, _ = load_libsvm_file(path.join(data_dir, file_path['Y']))
        return X, Y
    elif file_format == 'jsonlines':
        return load_json_lines_file(path.join(data_dir, file_path), features_fields=dataset_meta['features_fields'], labels_field=dataset_meta['labels_field'])
    else:
        raise ValueError("File format {} is not supported".format(file_format))


def to_csr_matrix(X, shape=None, sort_indices=False, dtype=np.float32):
    """
    Converts sparse matrix-like data, like list of list of tuples (idx, value), to Scipy csr_matrix.

    :param X: Matrix-like object to convert to csr_matrix: ndarray or list of lists of ints or tuples of ints and floats (idx, value).
    :type X: ndarray, list[list[int|str]], list[list[tuple[int, float]]
    :param shape: Shape of the matrix, if None, shape will be deduce from X, defaults to None
    :type shape: tuple, optional
    :param sort_indices: Sort rows' data by indices (idx), defaults to False
    :type sort_indices: bool, optional
    :param dtype: Data type of the matrix, defaults to np.float32
    :type dtype: type, optional
    :return: X as csr_matrix.
    :rtype: csr_matrix
    """
    if isinstance(X, list) and isinstance(X[0], (list, tuple, set)):
        first_element = _get_first_element_of_list_of_lists(X)

        if dtype is None:
            dtype = _get_dtype(first_element)

        size = 0
        for x in X:
            size += len(x)

        indptr = np.zeros(len(X) + 1, dtype=np.int32)
        indices = np.zeros(size, dtype=np.int32)
        data = np.ones(size, dtype=dtype)
        cells = 0

        if isinstance(first_element, tuple):
            for row, x in enumerate(X):
                indptr[row] = cells
                x = sorted(x) if sort_indices else x
                for x_i in x:
                    indices[cells] = x_i[0]
                    data[cells] = x_i[1]
                    cells += 1
            indptr[len(X)] = cells

        else:
            for row, x in enumerate(X):
                indptr[row] = cells
                indices[cells:cells + len(x)] = sorted(x) if sort_indices else x
                cells += len(x)
            indptr[len(X)] = cells

        array = csr_matrix((data, indices, indptr), shape=shape)
    else: # Try to convert via constructor
        array = csr_matrix(X, dtype=dtype, shape=shape)
    # raise TypeError('Cannot convert X to csr_matrix')

    # Check type
    if array.dtype != dtype:
        print("Conversion", array.dtype, dtype)
        array = array.astype(dtype)

    return array

def to_np_matrix(X, shape=None, dtype=np.float32):
    """
    Converts sparse matrix-like data, like list of list of tuples (idx, value), to Numpy matrix (2D array).

    :param X: Matrix-like object to convert to csr_matrix: ndarray or list of lists of ints or tuples of ints and floats (idx, value).
    :type X: ndarray, list[list[int|str]], list[list[tuple[int, float]]
    :param shape: Shape of the matrix, if None, shape will be deduce from X, defaults to None
    :type shape: tuple, optional
    :param dtype: Data type of the matrix, if None, type will be deduced from data, defaults to np.float32
    :type dtype: type, optional
    :return: X as Numpy 2D array.
    :rtype: ndarray
    """
    if isinstance(X, list) and isinstance(X[0], (list, tuple, set)):
        first_element = _get_first_element_of_list_of_lists(X)

        if dtype is None:
            dtype = _get_dtype(first_element)

        if shape is None:
            m = max([max(x) for x in X if len(x)]) + 1
            n = len(X)
            shape = (n, m)
        array = np.zeros(shape, dtype=dtype)

        if isinstance(first_element, (list, tuple)):
            for row, x in enumerate(X):
                for x_i in x:
                    array[row, x_i[0]] = x_i[1]
        else:
            for row, x in enumerate(X):
                array[row, x] = 1

    elif isinstance(X, csr_matrix):
        array = X.toarray()
    else: # Try to convert via constructor
        array = np.array(X, dtype=dtype)

    # Check type
    if array.dtype != dtype:
        print("Conversion", array.dtype, dtype)
        array = array.astype(dtype)

    return array


# Helpers
def _get_dtype(first_element):
    if isinstance(first_element, (list, tuple)):
        return type(first_element[1])
    else:
        return type(first_element)


def _get_first_element_of_list_of_lists(X):
    first_element = None
    for x in X:
        if len(x):
            first_element = x[0]
    if first_element is None:
        raise ValueError('X is does not contain any element')


def _get_data_meta(dataset, subset='train', format='bow'):
    # Handel aliases
    aliases = {
        'wiki10': 'wiki10-31k',
        'deliciouslarge': 'deliciouslarge-200k',
        'wikilshtc': 'wikilshtc-325k',
        'wikipedialarge': 'wikipedialarge-500k'
    }
    _dataset = dataset.lower()
    if _dataset in aliases:
        _dataset = aliases[_dataset]

    if _dataset not in DATASETS:
        raise ValueError("Dataset {} is not available".format(dataset))

    # Handel format aliases and versions
    _format = format
    if _format == 'tf-idf':
        _format = 'bow'
    if _format.startswith("bow") and _format not in DATASETS[_dataset]:
        _format = 'bow'

    if _format not in DATASETS[_dataset]['formats']:
        raise ValueError("Format {} is not available for dataset {}".format(format, dataset))

    if subset is not None and subset not in DATASETS[_dataset]['subsets']:
        raise ValueError("Subset {} is not available for dataset {}".format(format, dataset))

    return DATASETS[_dataset][_format]


def _download_file_from_google_drive(url, dest_path, overwrite=False, unzip=False, delete_zip=False, verbose=False):
    """
    Downloads a shared file from google drive into a given folder and optionally unzips it.

    :param url: File url to download
    :type url: str
    :param dest_path: The destination where to save the downloaded file
    :type dest_path: str
    :param overwrite: If True force redownload and overwrite, defaults to False
    :type overwrite: bool
    :param unzip: If True unzip a file, optional, defaults to False
    :type unzip: bool
    :param delete_zip: If True and unzips is True delete archive file after unziping, defaults to False
    :type delete_zip: bool
    :param verbose: If True print downloading progress, defaults to False
    :type verbose: bool
    :return: None
    """

    destination_directory = path.dirname(dest_path)
    if not path.exists(destination_directory):
        makedirs(destination_directory)

    if not path.exists(dest_path) or overwrite:
        if verbose:
            print('Downloading {} into {} ... '.format(url, dest_path))

        gdown.download(url=url, output=dest_path, quiet=not verbose)

        if unzip:
            try:
                if verbose:
                    print('Unzipping ...')

                with zipfile.ZipFile(dest_path, 'r') as z:
                    z.extractall(destination_directory)
                if delete_zip:
                    remove(dest_path)

            except zipfile.BadZipfile:
                warnings.warn('Ignoring `unzip` since "{}" does not look like a valid zip file'.format(dest_path))

        if verbose:
            print('Done.')

