import h5py
from pathlib import Path
import torch
from torch.utils import data


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).

    Example:
        num_epochs = 50
        loader_params = {'batch_size': 10, 'shuffle': True, 'num_workers': 6}
        dataset = HDF5Dataset('C:/ml/data', recursive=True, load_data=False,
           data_cache_size=4, transform=None)

        data_loader = data.DataLoader(dataset, **loader_params)
        for i in range(num_epochs):
           for x in data_loader:
    """

    def __init__(self, file_path, recursive, transform=None):
        super().__init__()
        self.data_info = []
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert (p.is_dir())
        if recursive:
            self.files = sorted(p.glob('**/*.h5'))
        else:
            self.files = sorted(p.glob('*.h5'))
        if len(self.files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in self.files:
            self._add_data_infos(str(h5dataset_fp.resolve()))

    def __getitem__(self, index):
        # get data
        x = self.get_data("array", index)

        return torch.from_numpy(x)

    def __len__(self):
        return len(self.files)

    def _add_data_infos(self, file_path):
        with h5py.File(file_path, 'r') as h5_file:
            # Walk through all groups, extracting datasets
            for dname, ds in h5_file.items():
                # for dname, ds in group.items():
                self.data_info.append(
                    {'file_path': file_path, 'type': dname,
                     'shape': (288, 495, 436, 3)})

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        with h5py.File(fp, 'r') as h5_file:
            data = h5_file[type][()]

        return data