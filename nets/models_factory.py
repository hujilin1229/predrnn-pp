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

    heading_table = tf.Tensor([[0, 0], [-1, 1], [1, 1], [-1, -1], [1, -1]], shape=(5, 2), dtype=tf.float32)
    func = networks_map[name]
    gt_images = []
    pred_images = []
    losses = []
    for i in range(1, 5):
        # tem_data = images.copy()
        heading_image = images[:, :, :, :, 2] * 255
        # print("Heading Unique", np.unique(heading_image), flush=True) #[  0.   1.  85. 170. 255.] output
        heading_image = tf.cast(heading_image // 85, tf.int8) + 1
        heading_image = tf.where(images[:, :, :, :, 2] == 0, tf.zeros_like(heading_image, tf.int8), heading_image)
        # heading_image[images[:, :, :, :, 2] == 0] = 0
        # print("Heading Unique", np.unique(heading_image), flush=True)
        # select the corresponding data
        heading_selected = tf.where(heading_image == i, heading_image, tf.zeros_like(heading_image, tf.int8))

        heading_image = heading_table[heading_selected]

        speed_on_axis = np.expand_dims(images[:, :, :, :, 1] / tf.sqrt(2), axis=-1)
        imss = speed_on_axis * heading_image

        gen_images, loss = func(imss, mask_true, num_layers, num_hidden, filter_size,
                                stride, seq_length, input_length, tln, batch_size=batch_size)

        gt_images.append(heading_image)
        pred_images.append(gen_images)
        losses.append(loss)

    gts = tf.math.add_n(gt_images)
    preds = tf.math.add_n(pred_images)

    gt_speed = tf.sqrt(gts[..., 0] ** 2 + gts[..., 1] ** 2)
    gen_speed = tf.sqrt(preds[..., 0] ** 2 + preds[..., 1] ** 2)

    loss_all = tf.math.add_n(losses)
    loss_all += masked_mse_tf(gen_speed, gt_speed, null_val=0.0)
    loss_all += masked_mse_tf(preds, gt_images, null_val=0.0)

    return [preds, loss_all]
