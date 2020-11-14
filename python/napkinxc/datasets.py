# Copyright (c) 2020 by Marek Wydmuch
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
from scipy.sparse import csr_matrix
from ._napkinxc import _load_libsvm_file


# List of all available datasets
DATASETS = {
    'eurlex-4k': {
        'name': 'Eurlex-4K',
        'formats': ['tf-idf'],
        'subsets': ['train', 'test'],
        'tf-idf': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGU0VTR1pCejFpWjg',
            'train': 'Eurlex/eurlex_train.txt',
            'test': 'Eurlex/eurlex_test.txt',
            'file_format': 'libsvm',
        }
    },
    'amazoncat-13k': {
        'name': 'AmazonCat-13K',
        'formats': ['tf-idf'],
        'subsets': ['train', 'test'],
        'tf-idf': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGa2tMbVJGdDNSMGc',
            'train': 'AmazonCat/amazonCat_train.txt',
            'test': 'AmazonCat/amazonCat_test.txt',
            'file_format': 'libsvm',
        }
    },
    'amazoncat-14k': {
        'name': 'AmazonCat-14K',
        'formats': ['tf-idf'],
        'subsets': ['train', 'test'],
        'tf-idf': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGaDFqU2E5U0dxS00',
            'train': 'AmazonCat-14K/amazonCat-14K_train.txt',
            'test': 'AmazonCat-14K/amazonCat-14K_test.txt',
            'file_format': 'libsvm',
        }
    },
    'wiki10-31k': {
        'name': 'Wiki10-31K',
        'formats': ['tf-idf'],
        'subsets': ['train', 'test'],
        'tf-idf': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGaDdOeGliWF9EOTA',
            'train': 'Wiki10/wiki10_train.txt',
            'test': 'Wiki10/wiki10_test.txt',
            'file_format': 'libsvm',
        }
    },
    'deliciouslarge-200k': {
        'name': 'DeliciousLarge-200K',
        'formats': ['tf-idf'],
        'subsets': ['train', 'test'],
        'tf-idf': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGR3lBWWYyVlhDLWM',
            'train': 'DeliciousLarge/deliciousLarge_train.txt',
            'test': 'DeliciousLarge/deliciousLarge_test.txt',
            'file_format': 'libsvm',
        }
    },
    'wikilshtc-325k': {
        'name': 'WikiLSHTC-325K',
        'formats': ['tf-idf'],
        'subsets': ['train', 'test'],
        'tf-idf': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGSHE1SWx4TVRva3c',
            'train': 'WikiLSHTC/wikiLSHTC_train.txt',
            'test': 'WikiLSHTC/wikiLSHTC_test.txt',
            'file_format': 'libsvm',
        }
    },
    'wikipedialarge-500k': {
        'name': 'WikipediaLarge-500K',
        'formats': ['tf-idf'],
        'subsets': ['train', 'test'],
        'tf-idf': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGRmEzVDVkNjBMR3c',
            'train': 'WikipediaLarge-500K/WikipediaLarge-500K_train.txt',
            'test': 'WikipediaLarge-500K/WikipediaLarge-500K_test.txt',
            'file_format': 'libsvm',
        }
    },
    'amazon-670k': {
        'name': 'Amazon-670K',
        'formats': ['tf-idf'],
        'subsets': ['train', 'test'],
        'tf-idf': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGdUJwRzltS1dvUVk',
            'train': 'Amazon/amazon_train.txt',
            'test': 'Amazon/amazon_test.txt',
            'file_format': 'libsvm',
        }
    },
    'amazon-3m': {
        'name': 'Amazon-3M',
        'formats': ['tf-idf'],
        'subsets': ['train', 'test'],
        'tf-idf': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGUEd4eTRxaWl3YkE',
            'train': 'Amazon-3M/amazon-3M_train.txt',
            'test': 'Amazon-3M/amazon-3M_test.txt',
            'file_format': 'libsvm',
        }
    },
}

# Create datasets aliases
DATASETS['wiki10'] = DATASETS['wiki10-31k']
DATASETS['deliciouslarge'] = DATASETS['deliciouslarge-200k']
DATASETS['wikilshtc'] = DATASETS['wikilshtc-325k']
DATASETS['wikipedialarge'] = DATASETS['wikipedialarge-500k']


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

    :param file: path to a file to load
    :type file: str
    :return: (csr_matrix, list[list[int]]), tuple of features matrix and labels
    """
    labels, indptr, indices, data = _load_libsvm_file(file)
    return csr_matrix((data, indices, indptr)), labels


def download_dataset(dataset, subset='train', format='tf-idf', root='./data', verbose=False):
    """
    Downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again.

    :param dataset: name of the dataset to load, case insensitive {'Eurlex-4K', 'AmazonCat-13K', 'AmazonCat-14K', 'Wiki10-31K', 'DeliciousLarge-200K', 'WikiLSHTC-325K', 'WikipediaLarge-500K', 'Amazon-670K', 'Amazon-3M'}
    :type dataset: str
    :param subset: subset of dataset to load into features matrix and labels {'train', 'test', 'all'}, defaults to 'train'
    :type subset: str, optional
    :param format: format of dataset to load {'tf-idf'}, defaults to 'tf-idf'
    :type format: str, optional
    :param root: location of datasets directory, defaults to './data'
    :type root: str, optional
    :param verbose: if True print downloading and loading progress, defaults to False
    :type verbose: bool, optional
    """
    dataset_meta = _get_data_meta(dataset, subset=subset, format=format)
    dataset_dest = path.join(root, dataset.lower() + '_' + format + ".zip")
    file_path = path.join(root, dataset_meta[subset])
    if not path.exists(file_path):
        if 'drive.google.com' in dataset_meta['url']:
            _download_file_from_google_drive(dataset_meta['url'], dataset_dest, unzip=True, overwrite=True, delete_zip=True, verbose=verbose)


def load_dataset(dataset, subset='train', format='tf-idf', root='./data', verbose=False):
    """
    Downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again.
    Then loads requested datasets into features matrix and labels.

    :param dataset: name of the dataset to load, case insensitive {'Eurlex-4K', 'AmazonCat-13K', 'AmazonCat-14K', 'Wiki10-31K', 'DeliciousLarge-200K', 'WikiLSHTC-325K', 'WikipediaLarge-500K', 'Amazon-670K', 'Amazon-3M'}
    :type dataset: str
    :param subset: subset of dataset to load into features matrix and labels {'train', 'test', 'all'}, defaults to 'train'
    :type subset: str, optional
    :param format: format of dataset to load {'tf-idf'}, defaults to 'tf-idf'
    :type format: str, optional
    :param root: location of datasets directory, defaults to ./data
    :type root: str, optional
    :param verbose: if True print downloading and loading progress, defaults to False
    :type verbose: bool, optional
    :return: (csr_matrix, list[list[int]]), tuple of features matrix and labels
    """
    dataset_meta = _get_data_meta(dataset, subset=subset, format=format)
    file_path = path.join(root, dataset_meta[subset])
    download_dataset(dataset, subset=subset, format=format, root=root, verbose=verbose)

    return _load_file(file_path, dataset_meta['file_format'])


# Helpers
def _get_data_meta(dataset, subset='train', format='tf-idf'):
    dataset = dataset.lower()

    if dataset not in DATASETS:
        raise ValueError("Dataset {} is not available".format(dataset))

    if format not in DATASETS[dataset]['formats']:
        raise ValueError("Format {} is not available for dataset {}".format(format, dataset))

    if subset is not None and subset not in DATASETS[dataset]['subsets']:
        raise ValueError("Subset {} is not available for dataset {}".format(format, dataset))

    return DATASETS[dataset][format]


def _load_file(filepath, format):
    if format == 'libsvm':
        return load_libsvm_file(filepath)


def _download_file_from_google_drive(url, dest_path, overwrite=False, unzip=False, delete_zip=False, verbose=False):
    """
    Downloads a shared file from google drive into a given folder and optionally unzips it.
    :param url: file url to download
    :type url: str
    :param dest_path: the destination where to save the downloaded file
    :type dest_path: str
    :param overwrite: if True force redownload and overwrite, defaults to False
    :type overwrite: bool
    :param unzip: if True unzip a file, optional, defaults to False
    :type unzip: bool
    :param delete_zip: if True and unzips is True delete archive file after unziping, defaults to False
    :type delete_zip: bool
    :param verbose: if True print downloading progress, defaults to False
    :type verbose: bool
    :return: None
    """

    download_url = 'https://drive.google.com/uc?export=download'
    re_match = re.search('id=([\w\d]+)', url)
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
