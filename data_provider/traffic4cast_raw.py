import numpy as np
from data_provider.h5dataset import HDF5Dataset
from torch.utils import data

class InputHandle:
    def __init__(self, input_param, mode='train'):
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        # self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.num_files = input_param['num_files']
        self.seq_len = input_param['seq_len']
        self.horizon = input_param['horizon']
        # self.mode = mode
        # self.data = {}
        # self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 0
        self.current_output_length = 0

    def total(self):
        return len(self.data_loader)

    def begin(self, do_shuffle = True):
        # construct dataset
        cache = 6
        loader_params = {'batch_size': self.num_files, 'shuffle': do_shuffle, 'num_workers': 0}
        self.dataset = HDF5Dataset(self.paths, recursive=False, load_data=False, data_cache_size=cache)
        self.data_loader = data.DataLoader(self.dataset, **loader_params)
        self.current_position = 0

    def get_data_files(self):
        return [d['file_path'] for d in self.dataset.get_data_infos('array')]

    def next(self):
        self.current_position += 1
        if self.no_batch_left():
            return None

    def no_batch_left(self):
        if self.current_position >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        data = next(iter(self.data_loader))
        data = data.numpy().astype(self.input_data_type) / 255.0
        num_files, num_data, height, width, channel = data.shape
        num_batch_per_file = num_data // (self.seq_len + self.horizon)
        batch = data[:, :num_batch_per_file*(self.seq_len + self.horizon)].reshape(
            num_files * num_batch_per_file, (self.seq_len+self.horizon), height, width, -1)

        return batch

    def get_test_batch(self, indices):
        # num of batches: len(indices)
        data = next(iter(self.data_loader))
        data = data.squeeze(0).numpy().astype(self.input_data_type) / 255.0
        data = [data[i-self.seq_len:i+self.horizon] for i in indices]
        data = np.stack(data, axis=0)

        return data
