__author__ = 'yunbo'

import tensorflow as tf
from layers.GradientHighwayUnit import GHU as ghu
from layers.CausalLSTMCell import CausalLSTMCell as cslstm
FLAGS = tf.app.flags.FLAGS

epsilon = 1e-11

def squash(v_j, dim = -1):
    """
    :param v_j: (?, 1, 1, 1, Co, d)
    :param dim:
    :return:
    """
    vec_squared_norm = tf.reduce_sum(tf.square(v_j), dim, keep_dims=True)
    a_j =  vec_squared_norm / (1 + vec_squared_norm)
    scalar_factor = a_j / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * v_j  # element-wise
    return vec_squashed

def rnn(images, mask_true, num_layers, num_hidden, filter_size, stride=1,
        seq_length=20, input_length=10, tln=True, batch_size=None):

    gen_images = []
    lstm = []
    cell = []
    hidden = []
    shape = images.get_shape().as_list()
    output_channels = shape[-1]

    for i in range(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell = cslstm('lstm_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln,
                          batch_size=batch_size)
        lstm.append(new_cell)
        cell.append(None)
        hidden.append(None)

    gradient_highway = ghu('highway', filter_size, num_hidden[0], tln=tln)

    mem = None
    z_t = None

    for t in range(seq_length-1):
        reuse = bool(gen_images)
        with tf.variable_scope('predrnn_pp', reuse=reuse):
            if t < input_length:
                inputs = images[:,t, ...]
            else:
                inputs = mask_true[:,t-input_length, ...]*images[:,t, ...] + (1-mask_true[:,t-input_length, ...])*x_gen

            hidden[0], cell[0], mem = lstm[0](inputs, hidden[0], cell[0], mem)
            z_t = gradient_highway(hidden[0], z_t, batch_size)
            hidden[1], cell[1], mem = lstm[1](z_t, hidden[1], cell[1], mem)

            # The output hidden here is the results of tanh * tahnh, which falls into the range of [-1, 1]
            for i in range(2, num_layers):
                hidden[i], cell[i], mem = lstm[i](hidden[i-1], hidden[i], cell[i], mem)

            x_gen = tf.layers.conv2d(inputs=hidden[num_layers-1],
                                     filters=output_channels,
                                     kernel_size=1,
                                     strides=1,
                                     # activation=tf.nn.tanh,
                                     padding='same',
                                     name="back_to_pixel")

            # squash
            x_gen = tf.reshape(x_gen, [-1, FLAGS.img_height, FLAGS.img_width,
                                       FLAGS.patch_size_height*FLAGS.patch_size_width, FLAGS.img_channel])
            x_gen = squash(x_gen, dim=-1) # makes a unit vector
            x_gen = tf.reshape(x_gen, [-1, FLAGS.img_height, FLAGS.img_width,
                                       FLAGS.patch_size_height*FLAGS.patch_size_width*FLAGS.img_channel])
            gen_images.append(x_gen)

    gen_images = tf.stack(gen_images, axis=1)
    # [batch_size, seq_length, height, width, channels]
    # gen_images = tf.transpose(gen_images, [1,0,2,3,4])
    loss = tf.nn.l2_loss(gen_images - images[:,1:])

    # add mask to loss to evaluate on valid vectors
    mask = images[:, 1:, :, :, 0] > 0
    gen_images = gen_images[mask]
    gt_images = images[:,1:][mask]
    loss = tf.nn.l2_loss(gen_images, gt_images)

    #loss += tf.reduce_sum(tf.abs(gen_images - images[:,1:]))
    return [gen_images, loss]

