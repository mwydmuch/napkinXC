# Copyright (c) 2020-2021 by Marek Wydmuch
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

import requests
import zipfile
import warnings
import re
from sys import stdout
from os import makedirs, path, remove
import numpy as np
from scipy.sparse import csr_matrix
from ._napkinxc import _load_libsvm_file


# List of all available datasets
DATASETS = {
    'eurlex-4k': {
        'name': 'EURLex-4K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGU0VTR1pCejFpWjg', # XMLC repo url
            'train': 'Eurlex/eurlex_train.txt',
            'test': 'Eurlex/eurlex_test.txt',
            'file_format': 'libsvm',
        }
    },
    'eurlex-4.3k': {
        'name': 'EURLex-4.3K',
        'formats': ['bow'],
        'subsets': ['train', 'test', 'validation'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1b3mWgaKIAmc9Ae3E0QrokiIFA9Qj1K9r', # XMLC repo url
            'train': 'EURLex-4.3K/train.txt',
            'test': 'EURLex-4.3K/test.txt',
            'validation': 'EURLex-4.3K/validation.txt',
            'file_format': 'libsvm',
        }
    },
    'amazoncat-13k': {
        'name': 'AmazonCat-13K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGa2tMbVJGdDNSMGc', # XMLC repo url
            'train': 'AmazonCat/amazonCat_train.txt',
            'test': 'AmazonCat/amazonCat_test.txt',
            'file_format': 'libsvm',
        }
    },
    'amazoncat-14k': {
        'name': 'AmazonCat-14K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGaDFqU2E5U0dxS00', # XMLC repo url
            'train': 'AmazonCat-14K/amazonCat-14K_train.txt',
            'test': 'AmazonCat-14K/amazonCat-14K_test.txt',
            'file_format': 'libsvm',
        }
    },
    'wiki10-31k': {
        'name': 'Wiki10-31K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGaDdOeGliWF9EOTA', # XMLC repo url
            'train': 'Wiki10/wiki10_train.txt',
            'test': 'Wiki10/wiki10_test.txt',
            'file_format': 'libsvm',
        }
    },
    'deliciouslarge-200k': {
        'name': 'DeliciousLarge-200K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGR3lBWWYyVlhDLWM', # XMLC repo url
            'train': 'DeliciousLarge/deliciousLarge_train.txt',
            'test': 'DeliciousLarge/deliciousLarge_test.txt',
            'file_format': 'libsvm',
        }
    },
    'wikilshtc-325k': {
        'name': 'WikiLSHTC-325K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGSHE1SWx4TVRva3c', # XMLC repo url
            'train': 'WikiLSHTC/wikiLSHTC_train.txt',
            'test': 'WikiLSHTC/wikiLSHTC_test.txt',
            'file_format': 'libsvm',
        }
    },
    'wikiseealsotitles-350k': {
        'name': 'WikiSeeAlsoTitles-350K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1bHtiLVF5EFsVL3qyU7y5e3M-fYHvXsG9', # XMLC repo url
            'train': {'X': 'WikiSeeAlsoTitles-350K/trn_X_Xf.txt', 'Y': 'WikiSeeAlsoTitles-350K/trn_X_Y.txt'},
            'test': {'X': 'WikiSeeAlsoTitles-350K/tst_X_Xf.txt', 'Y': 'WikiSeeAlsoTitles-350K/tst_X_Y.txt'},
            'file_format': 'XY_sparse',
        }
    },
    'wikititles-500k': {
        'name': 'WikiTitles-500K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=11U4qDWKvsR6pCzLvY3APckx-R_ihyMih', # XMLC repo url
            'train': {'X': 'WikiTitles-500K/trn_X_Xf.txt', 'Y': 'WikiTitles-500K/trn_X_Y.txt'},
            'test': {'X': 'WikiTitles-500K/tst_X_Xf.txt', 'Y': 'WikiTitles-500K/tst_X_Y.txt'},
            'file_format': 'XY_sparse',
        }
    },
    'wikipedialarge-500k': {
        'name': 'WikipediaLarge-500K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGRmEzVDVkNjBMR3c', # XMLC repo url
            'train': 'WikipediaLarge-500K/WikipediaLarge-500K_train.txt',
            'test': 'WikipediaLarge-500K/WikipediaLarge-500K_test.txt',
            'file_format': 'libsvm',
        }
    },
    'amazontitles-670k': {
        'name': 'AmazonTitles-670K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1OKnaLu4SDMOQ69rHdwF8ExKkeF2SZw7z', # XMLC repo url
            'train': {'X': 'AmazonTitles-670K/trn_X_Xf.txt', 'Y': 'AmazonTitles-670K/trn_X_Y.txt'},
            'test': {'X': 'AmazonTitles-670K/tst_X_Xf.txt', 'Y': 'AmazonTitles-670K/tst_X_Y.txt'},
            'file_format': 'XY_sparse',
        }
    },
    'amazon-670k': {
        'name': 'Amazon-670K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGdUJwRzltS1dvUVk', # XMLC repo url
            'train': 'Amazon/amazon_train.txt',
            'test': 'Amazon/amazon_test.txt',
            'file_format': 'libsvm',
        }
    },
    'amazontitles-3m': {
        'name': 'AmazonTitles-3M',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1PGzippnzIcgVNYZ8qQKVb0GNiARjQxvV', # XMLC repo url
            'train': {'X': 'AmazonTitles-3M/trn_X_Xf.txt', 'Y': 'AmazonTitles-3M/trn_X_Y.txt'},
            'test': {'X': 'AmazonTitles-3M/tst_X_Xf.txt', 'Y': 'AmazonTitles-3M/tst_X_Y.txt'},
            'file_format': 'XY_sparse',
        }
    },
    'amazon-3m': {
        'name': 'Amazon-3M',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGUEd4eTRxaWl3YkE', # XMLC repo url
            'train': 'Amazon-3M/amazon-3M_train.txt',
            'test': 'Amazon-3M/amazon-3M_test.txt',
            'file_format': 'libsvm',
        }
    },
    'lf-amazontitles-131k': {
        'name': 'LF-AmazonTitles-131K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1VlfcdJKJA99223fLEawRmrXhXpwjwJKn', # XMLC repo url
            'train': 'LF-AmazonTitles-131K/train.txt',
            'test': 'LF-AmazonTitles-131K/test.txt',
            'file_format': 'libsvm',
        }
    },
    'lf-amazon-131k': {
        'name': 'LF-Amazon-131K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1YNGEifTHu4qWBmCaLEBfjx07qRqw9DVW', # XMLC repo url
            'train': 'LF-Amazon-131K/train.txt',
            'test': 'LF-Amazon-131K/test.txt',
            'file_format': 'libsvm',
        }
    },
    'lf-wikiseealsotitles-320k': {
        'name': 'LF-WikiSeeAlsoTitles-320K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1edWtizAFBbUzxo9Z2wipGSEA9bfy5mdX', # XMLC repo url
            'train': 'LF-WikiSeeAlsoTitles-320K/train.txt',
            'test': 'LF-WikiSeeAlsoTitles-320K/test.txt',
            'file_format': 'libsvm',
        }
    },
    'lf-wikiseealso-320k': {
        'name': 'LF-WikiSeeAlso-320K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1N8C_RL71ErX6X92ew9h8qRuTWJ9LywE8', # XMLC repo url
            'train': 'LF-WikiSeeAlso-320K/train.txt',
            'test': 'LF-WikiSeeAlso-320K/test.txt',
            'file_format': 'libsvm',
        }
    },
    'lf-wikititles-500k': {
        'name': 'LF-WikiTitles-500K',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1qa1HTTD509J5r4yNAH-Aq6wArljgg4mx', # XMLC repo url
            'train': 'LF-WikiTitles-500K/train.txt',
            'test': 'LF-WikiTitles-500K/test.txt',
            'file_format': 'libsvm',
        }
    },
    'lf-amazontitles-1.3m': {
        'name': 'LF-AmazonTitles-1.3M',
        'formats': ['bow'],
        'subsets': ['train', 'test'],
        'bow': {
            'url': 'https://drive.google.com/uc?export=download&id=1Davc6BIfoTIAS3mP1mUY5EGcGr2zN2pO', # XMLC repo url
            'train': 'LF-AmazonTitles-1.3M/train.txt',
            'test': 'LF-AmazonTitles-1.3M/test.txt',
            'file_format': 'libsvm',
        }
    },
}


# Main functions for downloading and loading datasets
def load_libsvm_file(file):
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
    :return: (csr_matrix, list[list[int]]) - tuple of features matrix and labels
    """
    labels, indptr, indices, data = _load_libsvm_file(file)
    return csr_matrix((data, indices, indptr)), labels


def download_dataset(dataset, subset='train', format='bow', root='./data', verbose=False):
    """
    Downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again.

    :param dataset: Name of the dataset to load, case insensitive, available datasets:

        - ``'Eurlex-4K'``,
        - ``'Eurlex-4.3K'``,
        - ``'AmazonCat-13K'``,
        - ``'AmazonCat-14K'``,
        - ``'Wiki10-31K'`` (alias: ``'Wiki10'``),
        - ``'DeliciousLarge-200K'`` (alias: ``'DeliciousLarge'``)
        - ``'WikiLSHTC-325K'`` (alias: ``'WikiLSHTC'``)
        - ``'WikiSeeAlsoTitles-350K'``,
        - ``'WikiTitles-500K'``,
        - ``'WikipediaLarge-500K'`` (alias: ``'WikipediaLarge'``),
        - ``'AmazonTitles-670K'``,
        - ``'Amazon-670K'``,
        - ``'AmazonTitles-3M'``,
        - ``'Amazon-3M'``,
        - ``'LF-AmazonTitles-131K'``,
        - ``'LF-Amazon-131K'``,
        - ``'LF-WikiSeeAlsoTitles-320K'``,
        - ``'LF-WikiSeeAlso-320K'``,
        - ``'LF-WikiTitles-500K'``,
        - ``'LF-AmazonTitles-1.3M'``.

    :type dataset: str
    :param subset: Subset of dataset to load into features matrix and labels {``'train'``, ``'test'``, ``'validation'``}, defaults to ``'train'``
    :type subset: str, optional
    :param format: Format of dataset to load {``'bow'`` (bag-of-words/tf-idf weights, alias ``'tf-idf'``)}, defaults to ``'bow'``
    :type format: str, optional
    :param root: Location of datasets directory, defaults to ``'./data'``
    :type root: str, optional
    :param verbose: If True print downloading and loading progress, defaults to False
    :type verbose: bool, optional
    """
    dataset_meta = _get_data_meta(dataset, subset=subset, format=format)
    dataset_dest = path.join(root, dataset.lower() + '_' + format + ".zip")
    file_path = dataset_meta[subset]
    if isinstance(file_path, str):
        file_path = [file_path]
    elif isinstance(file_path, dict):
        file_path = file_path.values()
    if not all([path.exists(path.join(root, f)) for f in file_path]):
        if 'drive.google.com' in dataset_meta['url']:
            _download_file_from_google_drive(dataset_meta['url'], dataset_dest, unzip=True, overwrite=True, delete_zip=True, verbose=verbose)


def load_dataset(dataset, subset='train', format='bow', root='./data', verbose=False):
    """
    Downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again.
    Then loads requested datasets into features matrix and labels.

    :param dataset: Name of the dataset to load, case insensitive, available datasets:

        - ``'Eurlex-4K'``,
        - ``'Eurlex-4.3K'``,
        - ``'AmazonCat-13K'``,
        - ``'AmazonCat-14K'``,
        - ``'Wiki10-31K'`` (alias: ``'Wiki10'``),
        - ``'DeliciousLarge-200K'`` (alias: ``'DeliciousLarge'``)
        - ``'WikiLSHTC-325K'`` (alias: ``'WikiLSHTC'``)
        - ``'WikiSeeAlsoTitles-350K'``,
        - ``'WikiTitles-500K'``,
        - ``'WikipediaLarge-500K'`` (alias: ``'WikipediaLarge'``),
        - ``'AmazonTitles-670K'``,
        - ``'Amazon-670K'``,
        - ``'AmazonTitles-3M'``,
        - ``'Amazon-3M'``,
        - ``'LF-AmazonTitles-131K'``,
        - ``'LF-Amazon-131K'``,
        - ``'LF-WikiSeeAlsoTitles-320K'``,
        - ``'LF-WikiSeeAlso-320K'``,
        - ``'LF-WikiTitles-500K'``,
        - ``'LF-AmazonTitles-1.3M'``.

    :type dataset: str
    :param subset: Subset of dataset to load into features matrix and labels {``'train'``, ``'test'``, ``'validation'``}, defaults to ``'train'``
    :type subset: str, optional
    :param format: Format of dataset to load {``'bow'`` (bag-of-words/tf-idf weights, alias ``'tf-idf'``)}, defaults to ``'bow'``
    :type format: str, optional
    :param root: Location of datasets directory, defaults to ``'./data'``
    :type root: str, optional
    :param verbose: If True print downloading and loading progress, defaults to False
    :type verbose: bool, optional
    :return: Tuple of features matrix and labels.
    :rtype: (csr_matrix, list[list[int]])
    """
    download_dataset(dataset, subset=subset, format=format, root=root, verbose=verbose)
    dataset_meta = _get_data_meta(dataset, subset=subset, format=format)
    file_format = dataset_meta['file_format']
    file_path = dataset_meta[subset]

    if file_format == 'libsvm':
        return load_libsvm_file(path.join(root, file_path))
    elif file_format == 'XY_sparse':
        X, _ = load_libsvm_file(path.join(root, file_path['X']))
        Y, _ = load_libsvm_file(path.join(root, file_path['Y']))
        return X, Y


def to_csr_matrix(X, shape=None, sort_indices=False, dtype=np.float32):
    """
    Converts matrix-like object to Scipy csr_matrix.

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
        size = 0
        for x in X:
            size += len(x)

        indptr = np.zeros(len(X) + 1, dtype=np.int32)
        indices = np.zeros(size, dtype=np.int32)
        data = np.ones(size, dtype=dtype)
        cells = 0

        if isinstance(X[0][0], int):
            for row, x in enumerate(X):
                indptr[row] = cells
                indices[cells:cells + len(x)] = sorted(x) if sort_indices else x
                cells += len(x)
            indptr[len(X)] = cells

        elif isinstance(X[0][0], tuple):
            for row, x in enumerate(X):
                indptr[row] = cells
                x = sorted(x) if sort_indices else x
                for x_i in x:
                    indices[cells] = x_i[0]
                    data[cells] = x_i[1]
                    cells += 1
            indptr[len(X)] = cells

        return csr_matrix((data, indices, indptr), shape=shape)
    elif isinstance(X, np.ndarray):
        return csr_matrix(X, dtype=dtype, shape=shape)
    else:
        raise TypeError('Cannot convert X to csr_matrix')


# Helpers
def _get_data_meta(dataset, subset='train', format='bow'):
    aliases = {
        'wiki10': 'wiki10-31k',
        'deliciouslarge': 'deliciouslarge-200k',
        'wikilshtc': 'wikilshtc-325k',
        'wikipedialarge': 'wikipedialarge-500k'
    }

    _dataset = dataset.lower()
    if _dataset in aliases:
        _dataset = aliases[_dataset]
    _format = format
    if _format == 'tf-idf':
        _format = 'bow'

    if _dataset not in DATASETS:
        raise ValueError("Dataset {} is not available".format(dataset))

    if _format not in DATASETS[_dataset]['formats']:
        raise ValueError("Format {} is not available for dataset {}".format(format, dataset))

    if subset is not None and subset not in DATASETS[_dataset]['subsets']:
        raise ValueError("Subset {} is not available for dataset {}".format(format, dataset))

    return DATASETS[_dataset][format]


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

    download_url = 'https://drive.google.com/uc?export=download'
    re_match = re.search('id=([\w\d\-]+)', url)
    file_id = re_match.group(1)

    destination_directory = path.dirname(dest_path)
    if not path.exists(destination_directory):
        makedirs(destination_directory)

    if not path.exists(dest_path) or overwrite:
        if verbose:
            print('Downloading {} into {} ... '.format(url, dest_path))

        session = requests.Session()
        response = session.get(download_url, params={'id': file_id}, stream=True)
        token = _get_google_drive_confirm_token(response)
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(download_url, params=params, stream=True)
        current_download_size = [0]
        _save_response_content(response, dest_path, verbose, current_download_size)

        if unzip:
            try:
                if verbose:
                    print('Unzipping ...')

                with zipfile.ZipFile(dest_path, 'r') as z:
                    z.extractall(destination_directory)
                if delete_zip:
                    remove(dest_path)

            except zipfile.BadZipfile:
                warnings.warn('Ignoring `unzip` since "{}" does not look like a valid zip file'.format(file_id))

        if verbose:
            print('Done.')


def _get_google_drive_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def _save_response_content(response, destination, verbose, current_size, chunk_size=32768):
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                if verbose:
                    print('\r' + _sizeof_fmt(current_size[0]), end=' ')
                    stdout.flush()
                    current_size[0] += chunk_size


def _sizeof_fmt(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return '{:.1f} {}{}'.format(num, unit, suffix)
        num /= 1024.0
    return '{:.1f} {}{}'.format(num, 'Y', suffix)
