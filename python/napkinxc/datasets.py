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
from sklearn.datasets import load_svmlight_file


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
            'load_params': {'multilabel': True, 'zero_based': True, 'n_features': 5000, 'offset': 1}
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
            'load_params': {'multilabel': True, 'zero_based': True, 'n_features': 203882, 'offset': 1}
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
            'load_params': {'multilabel': True, 'zero_based': True, 'n_features': 597540, 'offset': 1}
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
            'load_params': {'multilabel': True, 'zero_based': True, 'n_features': 101938, 'offset': 1}
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
            'load_params': {'multilabel': True, 'zero_based': True, 'n_features': 782585, 'offset': 1}
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
            'load_params': {'multilabel': True, 'zero_based': True, 'n_features': 1617899, 'offset': 1}
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
            'load_params': {'multilabel': True, 'zero_based': True, 'n_features': 2381304, 'offset': 1}
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
            'load_params': {'multilabel': True, 'zero_based': True, 'n_features': 135909, 'offset': 1}
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
            'load_params': {'multilabel': True, 'zero_based': True, 'n_features': 337067, 'offset': 1}
        }
    },
}

# Create datasets aliases
DATASETS['wiki10'] = DATASETS['wiki10-31k']
DATASETS['deliciouslarge'] = DATASETS['deliciouslarge-200k']
DATASETS['wikilshtc'] = DATASETS['wikilshtc-325k']
DATASETS['wikipedialarge'] = DATASETS['wikipedialarge-500k']


# Main function for downloading and loading datasets
def load_dataset(dataset, subset='train', format='tf-idf', root='./data', verbose=False):
    """
    Downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again.
    Then loads requested datasets into features matrix and labels.
    :param dataset: name of the dataset to load, case insensitive
        {'Eurlex-4K', 'AmazonCat-13K', 'AmazonCat-14K', 'Wiki10-31K', 'DeliciousLarge-200K', 'WikiLSHTC-325K',
         'WikipediaLarge-500K', 'Amazon-670K', 'Amazon-3M'}
    :type dataset: str
    :param subset: subset of dataset to load into features matrix and labels {'train', 'test', 'all'}, defaults to 'train'
    :type subset: str
    :param format: format of dataset to load {'tf-idf'}, defaults to 'tf-idf'
    :type format: str
    :param root: location of datasets directory, defaults to ./data
    :type root: str
    :param verbose: if True print downloading and loading progress, defaults to False
    :type verbose: bool
    :return: (csr_matrix, list[list[int]]), features matrix and labels
    """
    dataset = dataset.lower()
    dataset_meta = DATASETS.get(dataset, None)

    if dataset_meta is None:
        raise ValueError("Dataset {} is not avialable".format(dataset))

    if format not in dataset_meta['formats']:
        raise ValueError("Format {} is not available for dataset {}".format(format, dataset))

    if subset is not None and subset not in dataset_meta['subsets']:
        raise ValueError("Subset {} is not available for dataset {}".format(format, dataset))

    dataset_dest = path.join(root, dataset_meta['name'] + '_' + format + ".zip")
    format_meta = dataset_meta[format]
    train_path = path.join(root, format_meta['train'])
    test_path = path.join(root, format_meta['test'])

    if not path.exists(train_path) or not path.exists(test_path):
        if 'drive.google.com' in format_meta['url']:
            GoogleDriveDownloader.download_file(format_meta['url'], dataset_dest, unzip=True, overwrite=True, delete_zip=True, verbose=verbose)

    if subset == 'train':
        return _load_file(train_path, format_meta['file_format'], format_meta['load_params'])
    elif subset == 'test':
        return _load_file(test_path, format_meta['file_format'], format_meta['load_params'])


# Helpers
def _load_file(filepath, format, load_params):
    if format == 'libsvm':
        return load_svmlight_file(filepath, **load_params)


# Based on: https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
class GoogleDriveDownloader:
    """
    Minimal class to download shared files from Google Drive.
    """

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = 'https://drive.google.com/uc?export=download'

    @staticmethod
    def download_file(url, dest_path, overwrite=False, unzip=False, delete_zip=False, verbose=False):
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

        re_match = re.search('id=([\w\d]+)', url)
        file_id = re_match.group(1)

        destination_directory = path.dirname(dest_path)
        if not path.exists(destination_directory):
            makedirs(destination_directory)

        if not path.exists(dest_path) or overwrite:
            if verbose:
                print('Downloading {} into {} ... '.format(url, dest_path))

            session = requests.Session()
            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params={'id': file_id}, stream=True)
            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True)
            current_download_size = [0]
            GoogleDriveDownloader._save_response_content(response, dest_path, verbose, current_download_size)

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

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination, verbose, current_size):
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    if verbose:
                        print('\r' + GoogleDriveDownloader._sizeof_fmt(current_size[0]), end=' ')
                        stdout.flush()
                        current_size[0] += GoogleDriveDownloader.CHUNK_SIZE

    # From: https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    @staticmethod
    def _sizeof_fmt(num, suffix='B'):
        for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
            if abs(num) < 1024.0:
                return '{:.1f} {}{}'.format(num, unit, suffix)
            num /= 1024.0
        return '{:.1f} {}{}'.format(num, 'Y', suffix)
