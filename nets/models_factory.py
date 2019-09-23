import tensorflow as tf
import numpy as np
from nets import predrnn_pp, predrnn_pp_capsule, predrnn_pp_capsule_multi_task
from nets.predrnn_pp_capsule_multi_task import masked_mse_tf

networks_map = {'predrnn_pp': predrnn_pp.rnn,
                'predrnn_pp_capsule': predrnn_pp_capsule.rnn,
                'predrnn_pp_capsule_multitask': predrnn_pp_capsule_multi_task.rnn
               }

def construct_model(name, images, mask_true, num_layers, num_hidden,
                    filter_size, stride, seq_length, input_length, tln, batch_size=None):
    '''Returns a sequence of generated frames
    Args:
        name: [predrnn_pp]
        mask_true: for schedualed sampling.
        num_hidden: number of units in a lstm layer.
        filter_size: for convolutions inside lstm.
        stride: for convolutions inside lstm.
        seq_length: including ins and outs.
        input_length: for inputs.
        tln: whether to apply tensor layer normalization.
    Returns:
        gen_images: a seq of frames.
        loss: [l2 / l1+l2].
    Raises:
        ValueError: If network `name` is not recognized.
    '''
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]
    return func(images, mask_true, num_layers, num_hidden, filter_size,
                stride, seq_length, input_length, tln, batch_size=batch_size)

def construct_multi_task_model(name, images, mask_true, num_layers, num_hidden,
                    filter_size, stride, seq_length, input_length, tln, batch_size=None):
    '''Returns a sequence of generated frames
    Args:
        name: [predrnn_pp]
        mask_true: for schedualed sampling.
        num_hidden: number of units in a lstm layer.
        filter_size: for convolutions inside lstm.
        stride: for convolutions inside lstm.
        seq_length: including ins and outs.
        input_length: for inputs.
        tln: whether to apply tensor layer normalization.
    Returns:
        gen_images: a seq of frames.
        loss: [l2 / l1+l2].
    Raises:
        ValueError: If network `name` is not recognized.
    '''
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)

    func = networks_map[name]
    gt_images = []
    pred_images = []
    losses = []
    original_shape = images.get_shape()
    shape1 = original_shape.as_list()
    shape1 = [-1] + shape1[1:-1] + [shape1[-1]//2, 2]
    images_dim2 = tf.reshape(images, shape1)
    for i in range(1, 5):
        if i == 1:
            pos = (images_dim2[..., 0] < 0) & (images_dim2[..., 1] > 0)
        elif i == 2:
            pos = (images_dim2[..., 0] > 0) & (images_dim2[..., 1] > 0)
        elif i == 3:
            pos = (images_dim2[..., 0] < 0) & (images_dim2[..., 1] < 0)
        else:
            pos = (images_dim2[..., 0] > 0) & (images_dim2[..., 1] < 0)
        pos = tf.expand_dims(pos, axis=-1)
        pos = tf.concat([pos, pos], axis=-1)

        imss = tf.where(pos, images_dim2, tf.zeros_like(images_dim2))

        gt_images.append(imss[:, 1:, ...])

        imss = tf.reshape(imss, [-1] + list(original_shape)[1:])
        gen_images, loss = func(imss, mask_true, num_layers, num_hidden, filter_size,
                                stride, seq_length, input_length, tln, batch_size=batch_size)

        pred_images.append(gen_images)
        losses.append(loss)

    gts = tf.math.add_n(gt_images)
    preds = tf.math.add_n(pred_images)

    # reshape back
    preds = tf.reshape(preds, shape1)
    gt_speed = tf.sqrt(gts[..., 0] ** 2 + gts[..., 1] ** 2)
    gen_speed = tf.sqrt(preds[..., 0] ** 2 + preds[..., 1] ** 2)

    loss_all = tf.math.add_n(losses)
    loss_all += masked_mse_tf(gen_speed, gt_speed, null_val=0.0)
    loss_all += masked_mse_tf(preds, gt_images, null_val=0.0)

    return [preds, loss_all]
