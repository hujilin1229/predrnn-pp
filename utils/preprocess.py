__author__ = 'yunbo'

import numpy as np
import os
import datetime
import re
import h5py
from pathlib import Path

def reshape_patch(img_tensor, patch_size_width, patch_size_height=None):
    if patch_size_height is None:
        patch_size_height = patch_size_width
    assert 5 == img_tensor.ndim
    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    img_height = np.shape(img_tensor)[2]
    img_width = np.shape(img_tensor)[3]
    num_channels = np.shape(img_tensor)[4]
    a = np.reshape(img_tensor, [batch_size, seq_length,
                                int(img_height/patch_size_height), patch_size_height,
                                int(img_width/patch_size_width), patch_size_width,
                                num_channels])

    b = np.transpose(a, [0,1,2,4,3,5,6])
    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  int(img_height/patch_size_height),
                                  int(img_width/patch_size_width),
                                  patch_size_width*patch_size_height*num_channels])
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size_width, patch_size_height=None):
    if patch_size_height is None:
        patch_size_height = patch_size_width

    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = int(channels / (patch_size_width*patch_size_height))
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size_height, patch_size_width,
                                  img_channels])
    b = np.transpose(a, [0,1,2,4,3,5,6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                int(patch_height * patch_size_height),
                                int(patch_width * patch_size_width),
                                img_channels])
    return img_tensor

def return_date(file_name):
    """Auxilliary function which returns datetime object from Traffic4Cast filename.

        Args.:
            file_name (str): file name, e.g., '20180516_100m_bins.h5'

        Returns: date string, e.g., '2018-05-16'
    """

    match = re.search(r'\d{4}\d{2}\d{2}', file_name)
    date = datetime.datetime.strptime(match.group(), '%Y%m%d').date()
    return date

def list_filenames(directory, excluded_dates=[]):
    """Auxilliary function which returns list of file names in directory in random order,
        filtered by excluded dates.

        Args.:
            directory (str): path to directory
            excluded_dates (list): list of dates which should not be included in result list,
                e.g., ['2018-01-01', '2018-12-31']

        Returns: list
    """
    filenames = os.listdir(directory)
    # np.random.shuffle(filenames)

    if len(excluded_dates) > 0:
        # check if in excluded dates
        excluded_dates = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in excluded_dates]
        filenames = [x for x in filenames if return_date(x) not in excluded_dates]

    return filenames

def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data = data, compression='gzip', compression_opts=9)
    f.close()

def create_directory_structure(root):
    berlin = os.path.join(root, "Berlin","Berlin_test")
    istanbul = os.path.join(root, "Istanbul","Istanbul_test")
    moscow = os.path.join(root, "Moscow", "Moscow_test")
    try:
        os.makedirs(berlin)
        os.makedirs(istanbul)
        os.makedirs(moscow)
    except OSError:
        print("failed to create directory structure")
        # sys.exit(2)

def construct_road_network_from_grid_condense(
        row_patch, col_patch, file_dir, least_ratio=0.033):

    print("1. Query a nodes from the validation data folder ")
    # Search for all h5 files
    p = Path(file_dir)
    assert (p.is_dir())
    files = p.glob('*.h5')
    data_all = []
    for h5dataset_fp in files:
        file_path = str(h5dataset_fp.resolve())
        with h5py.File(file_path, 'r') as f:
            data = f['array'][()]
            data_all.append(data)
    data_all = np.stack(data_all, axis=0)
    batch, timeslots, rows, cols, num_channels = data_all.shape
    data_patch = np.reshape(data_all, (batch, timeslots, rows//row_patch, row_patch,
                                       cols//col_patch, col_patch, num_channels))
    non_zeros = np.sum(data_patch > 0, axis=(0, 1, 3, 5, 6))
    total_num_counts = batch * timeslots * row_patch * col_patch * num_channels
    non_zeros_x, non_zeros_y = np.nonzero(non_zeros > total_num_counts * least_ratio)
    node_pos = np.stack([non_zeros_x, non_zeros_y], axis=1)

    return node_pos