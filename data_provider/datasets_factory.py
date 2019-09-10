from data_provider import mnist, traffic4cast_raw
import h5py
import numpy as np
# import cv2

datasets_map = {
    'mnist': mnist,
    'Berlin': traffic4cast_raw,
    'Istanbul': traffic4cast_raw,
    'Moscow': traffic4cast_raw
}

def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  is_training=True, seq_len=12, horizon=3):
    '''Given a dataset name and returns a Dataset.
    Args:
        dataset_name: String, the name of the dataset.
        train_data_paths: List, [train_data_path1, train_data_path2...]
        valid_data_paths: List, [val_data_path1, val_data_path2...]
        batch_size: Int
        is_training: Bool
    Returns:
        if is_training:
            Two dataset instances for both training and evaluation.
        else:
            One dataset instance for evaluation.
    Raises:
        ValueError: If `dataset_name` is unknown.
    '''

    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')

    if dataset_name == 'mnist':
        test_input_param = {'paths': valid_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name+'test iterator'}
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle = False)
        if is_training:
            train_input_param = {'paths': train_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name+' train iterator'}
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)
            train_input_handle.begin(do_shuffle = True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle
    else:
        test_input_param = {'paths': valid_data_paths,
                            'num_files': batch_size,
                            'input_data_type': 'float32',
                            'output_data_type': 'float32',
                            'seq_len': seq_len,
                            'horizon': horizon
                            }
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param, 'valid')
        test_input_handle.begin(do_shuffle = False)
        if is_training:
            train_input_param = {'paths': train_data_paths,
                                 'num_files': batch_size,
                                 'input_data_type': 'float32',
                                 'output_data_type': 'float32',
                                 'seq_len': seq_len,
                                 'horizon': horizon
                                 }
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param, 'train')
            train_input_handle.begin(do_shuffle = True)

            return train_input_handle, test_input_handle
        else:
            return test_input_handle

def test_validation_provider(data_file, indices, down_sample=4, seq_len=12, horizon=3):

    try:
        fr = h5py.File(data_file, 'r')
        data = fr.get('array').value
        fr.close()
    except:
        return None

    input_raw_data = [[], [], []]
    data = np.transpose(data, (0, 3, 1, 2))
    for j in range(data.shape[1]):
        for i in range(data.shape[0]):
            tmp_data = data[i, j, :, :]
            n_rows, n_cols = tmp_data.shape
            # down sample the image
            # tmp_data = cv2.resize(tmp_data, (n_cols // down_sample, n_rows // down_sample))
            input_raw_data[j].append(tmp_data)

    # stack volume, speed and heading together
    input_raw_data_channel_1 = np.stack(input_raw_data[0], axis=0)  # volume
    input_raw_data_channel_2 = np.stack(input_raw_data[1], axis=0)  # speed
    input_raw_data_channel_3 = np.stack(input_raw_data[2], axis=0)  # heading

    # expand dims on axis1
    input_raw_data_channel_1 = np.expand_dims(input_raw_data_channel_1, axis=1)
    input_raw_data_channel_2 = np.expand_dims(input_raw_data_channel_2, axis=1)
    input_raw_data_channel_3 = np.expand_dims(input_raw_data_channel_3, axis=1)

    input_raw_data_channel_1 = [input_raw_data_channel_1[i:i + seq_len + horizon] for i in indices]
    input_raw_data_channel_2 = [input_raw_data_channel_2[i:i + seq_len + horizon] for i in indices]
    input_raw_data_channel_3 = [input_raw_data_channel_3[i:i + seq_len + horizon] for i in indices]

    original_output_data = [data[i+seq_len:i+seq_len+horizon] for i in indices]

    input_raw_data_channel_2 = np.stack(input_raw_data_channel_2, axis=0)
    original_output_data = np.stack(original_output_data, axis=0)
    original_output_data = np.transpose(original_output_data, [0, 1, 3, 4, 2])
    input_raw_data_channel_2 = np.transpose(input_raw_data_channel_2, [0, 1, 3, 4, 2])

    return input_raw_data_channel_2, original_output_data
