from data_provider import mnist, traffic4cast

datasets_map = {
    'mnist': mnist,
    'Berlin': traffic4cast,
    'Istanbul': traffic4cast,
    'Moscow': traffic4cast
}

def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  is_training=True, down_sample=4, seq_len=12, horizon=3):
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
        test_input_param = {'paths': valid_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name+'_validation',
                            'down_sample': down_sample,
                            'seq_len': seq_len,
                            'horizon': horizon
                            }
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param, 'valid')
        test_input_handle.begin(do_shuffle = False)
        if is_training:
            train_input_param = {'paths': train_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name+'_training',
                                 'down_sample': down_sample,
                                 'seq_len': seq_len,
                                 'horizon': horizon
                                 }
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param, 'train')
            train_input_handle.begin(do_shuffle = True)

            return train_input_handle, test_input_handle
        else:
            return test_input_handle
