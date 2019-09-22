# this file is adapted from train.py which is authored by 'yunbo'
__author__ = 'jilin'

import os.path
import numpy as np
import tensorflow as tf
from nets import models_factory
from data_provider import datasets_factory
from utils import preprocess
from utils import metrics
import h5py

def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        # mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)

        return np.sum(rmse) / np.sum(mask)

def cast_moving_avg(data):
    """
    Returns cast moving average (cast to np.uint8)
    data = tensor of shape (5, step, 495, 436, 3) of  type float32
    Return: tensor of shape (5, 3, 495, 436, 3) of type uint8
    """

    prediction = []
    for i in range(3):
        data_slice = data[:, i:]
        # print(i, data_slice.shape)
        # sol 1. only avg_mean without considering empty slots: looks okay with mse: 4114
        t = np.mean(data_slice, axis = 1)
        t = np.expand_dims(t, axis=1)
        prediction.append(t)
        data = np.concatenate([data, t], axis =1)

    prediction = np.concatenate(prediction, axis = 1)

    return prediction

# -----------------------------------------------------------------------------
FLAGS = tf.app.flags.FLAGS

# data I/O
tf.app.flags.DEFINE_string('dataset_name', 'Berlin', #'mnist',
                           'The name of dataset.')
tf.app.flags.DEFINE_string('train_data_paths',
                           './data/', # 'data/moving-mnist-example/moving-mnist-train.npz',
                           'train data paths.')
tf.app.flags.DEFINE_string('valid_data_paths',
                           './data/', # 'data/moving-mnist-example/moving-mnist-valid.npz',
                           'validation data paths.')
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/predrnn_pp_capsule',
                            'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_frm_dir', 'results/predrnn_pp_capsule',
                           'dir to store result.')
# model
tf.app.flags.DEFINE_string('model_name', 'predrnn_pp_capsule',
                           'The name of the architecture.')
tf.app.flags.DEFINE_string('pretrained_model', '',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('input_length', 6,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('seq_length', 9,
                            'total input and output length.')
tf.app.flags.DEFINE_integer('batch_size', 16,
                            'total input and output length.')
tf.app.flags.DEFINE_integer('img_width', 109, # 435 /5
                            'input image width.')
tf.app.flags.DEFINE_integer('img_height', 99, # 496/4
                            'input image width.')
tf.app.flags.DEFINE_integer('img_channel', 2,
                            'number of image channel.')
tf.app.flags.DEFINE_integer('stride', 1,
                            'stride of a convlstm layer.')
tf.app.flags.DEFINE_integer('filter_size', 5,
                            'filter of a convlstm layer.')
tf.app.flags.DEFINE_string('num_hidden', '32,32',
                           'COMMA separated number of units in a convlstm layer.')
tf.app.flags.DEFINE_integer('patch_size_height', 5,
                            'patch size on one dimension.')
tf.app.flags.DEFINE_integer('patch_size_width', 4,
                            'patch size on one dimension.')
tf.app.flags.DEFINE_integer('heading', 1,
                            'the select heading.')

tf.app.flags.DEFINE_boolean('layer_norm', True,
                            'whether to apply tensor layer norm.')
# optimization
tf.app.flags.DEFINE_float('lr', 0.001,
                          'base learning rate.')
tf.app.flags.DEFINE_boolean('reverse_input', True,
                            'whether to reverse the input frames while training.')
tf.app.flags.DEFINE_integer('batch_file', 1,
                            'num of file per batch for training.')
tf.app.flags.DEFINE_integer('max_iterations', 28500,
                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 10,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 10,
                            'number of iters for test.')
tf.app.flags.DEFINE_integer('snapshot_interval', 10,
                            'number of iters saving models.')

tf.app.flags.DEFINE_string('mode', 'validation',
                           'COMMA separated number of units in a convlstm layer.')
tf.app.flags.DEFINE_integer('test_time', 16,
                           'COMMA separated number of units in a convlstm layer.')
tf.app.flags.DEFINE_string('loss_func', 'L1+L2+VALID',
                           'the applied loss function "L1+L2+VALID" or "L1+L2+ALL" or "L1+VALID" or "L1+VALID".')

class Model(object):
    def __init__(self):
        # inputs
        self.x = tf.placeholder(tf.float32,
                                [None,
                                 FLAGS.seq_length,
                                 FLAGS.img_height,
                                 FLAGS.img_width,
                                 int(FLAGS.patch_size_height*FLAGS.patch_size_width*FLAGS.img_channel)])

        self.mask_true = tf.placeholder(tf.float32,
                                        [None,
                                         FLAGS.seq_length-FLAGS.input_length-1,
                                         FLAGS.img_height,
                                         FLAGS.img_width,
                                         int(FLAGS.patch_size_height*FLAGS.patch_size_width*FLAGS.img_channel)])
        self.batchsize = tf.placeholder(tf.int32, [], name='batchsize')

        grads = []
        self.pred_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        num_hidden = [int(x) for x in FLAGS.num_hidden.split(',')]
        print("hidden shape is ", num_hidden, flush=True)
        num_layers = len(num_hidden)
        with tf.variable_scope(tf.get_variable_scope()):
            # define a model
            output_list = models_factory.construct_model(
                FLAGS.model_name, self.x,
                self.mask_true,
                num_layers, num_hidden,
                FLAGS.filter_size, FLAGS.stride,
                FLAGS.seq_length, FLAGS.input_length,
                FLAGS.layer_norm,
                self.batchsize)
            gen_ims = output_list[0]
            loss = output_list[1]
            pred_ims = gen_ims[:,FLAGS.input_length-1:]
            self.loss_train = loss
            # gradients
            all_params = tf.trainable_variables()
            grads.append(tf.gradients(loss, all_params))
            self.pred_seq.append(pred_ims)

        self.train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config = configProt)
        self.sess.run(init)
        if FLAGS.pretrained_model:
            try:
                self.saver.restore(self.sess, tf.train.latest_checkpoint(FLAGS.pretrained_model))
            except:
                pass

    def train(self, inputs, lr, mask_true, batch_size):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.tf_lr: lr})
        feed_dict.update({self.mask_true: mask_true})
        feed_dict.update({self.batchsize: batch_size})
        loss, _, pred_seq_list = self.sess.run((self.loss_train, self.train_op, self.pred_seq), feed_dict)
        return loss, pred_seq_list

    def test(self, inputs, mask_true, batch_size):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.mask_true: mask_true})
        feed_dict.update({self.batchsize: batch_size})
        gen_ims = self.sess.run(self.pred_seq, feed_dict)
        return gen_ims

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir, flush=True)

def main(argv=None):

    # FLAGS.save_dir += FLAGS.dataset_name + str(FLAGS.seq_length) + FLAGS.num_hidden + 'squash'
    heading_dict = {1: 1, 2: 85, 3: 170, 4: 255, 0: 0}
    heading = FLAGS.heading
    loss_func = 'L1+L2+VALID'
    # FLAGS.save_dir += FLAGS.dataset_name + str(FLAGS.seq_length) + FLAGS.num_hidden + 'squash' + loss_func + str(
    #     heading)
    # FLAGS.pretrained_model = FLAGS.save_dir

    test_data_paths = os.path.join(FLAGS.valid_data_paths, FLAGS.dataset_name, FLAGS.dataset_name + '_' + FLAGS.mode)
    sub_files = preprocess.list_filenames(test_data_paths, [])

    output_path = f'../Results/t{FLAGS.test_time}_{FLAGS.mode}/{FLAGS.loss_func}'
    # output_path = f'./Results/predrnn/t14/'
    # preprocess.create_directory_structure(output_path)

    # The following indicies are the start indicies of the 3 images to predict in the 288 time bins (0 to 287)
    # in each daily test file. These are time zone dependent. Berlin lies in UTC+2 whereas Istanbul and Moscow
    # lie in UTC+3.
    utcPlus2 = [30, 69, 126, 186, 234]
    utcPlus3 = [57, 114, 174, 222, 258]
    heading_table = np.array([[0, 0], [-1, 1], [1, 1], [-1, -1], [1, -1]], dtype=np.float32)

    indicies = utcPlus3
    if FLAGS.dataset_name == 'Berlin':
        indicies = utcPlus2

    epsilon = 0.2 * 255
    capsule_combine_list = []
    gt_list = []
    mavg_list = []
    for f in sub_files:
        with h5py.File(os.path.join(test_data_paths, f), 'r') as h5_file:
            data = h5_file['array'][()]
            # get relevant training data pieces
            data = [data[y - FLAGS.input_length:y + FLAGS.seq_length - FLAGS.input_length] for y in indicies]
            test_ims = np.stack(data, axis=0)
            gt = test_ims[:, FLAGS.input_length:, :, :, :]
            capsule_combine = np.zeros_like(test_ims[:, FLAGS.input_length:, :, :, :2], dtype=np.float32)

            mavg_results = cast_moving_avg(test_ims[:, :FLAGS.input_length, ...])
            for heading_i in range(1, 5):
                outfile = os.path.join(output_path, FLAGS.dataset_name, FLAGS.dataset_name + '_test', f'{heading_i}' + f)
                with h5py.File(outfile, 'r') as heading_file:
                    capsule_i = heading_file['array'][()]
                    capsule_combine += capsule_i

                    gt_head_pos = gt[..., 2] == heading_dict[heading_i]
                    gt_head = np.zeros_like(gt)
                    gt_head[gt_head_pos] = gt[gt_head_pos]

                    val_results_speed = np.sqrt(
                        capsule_i[..., 0] ** 2 + capsule_i[..., 1] ** 2) * 255.0

                    # print("val speed: ", val_results_speed, flush=True)
                    val_results_heading = np.zeros_like(capsule_i[..., 1])
                    val_results_heading[(capsule_i[..., 0] > 0) & (capsule_i[..., 1] > 0)] = 85.0
                    val_results_heading[(capsule_i[..., 0] > 0) & (capsule_i[..., 1] < 0)] = 255.0
                    val_results_heading[(capsule_i[..., 0] < 0) & (capsule_i[..., 1] < 0)] = 170.0
                    val_results_heading[(capsule_i[..., 0] < 0) & (capsule_i[..., 1] > 0)] = 1.0

                    # val_results_heading[mavg_results[:, :, :, :, 1] < epsilon] = \
                    #     mavg_results[:, :, :, :, 2][mavg_results[:, :, :, :, 1] < epsilon]

                    gen_speed_heading = np.stack([val_results_speed, val_results_heading], axis=-1)

                    print("Heading ", heading_i)
                    print("MSE on all pixels")
                    mse = masked_mse_np(gen_speed_heading, gt_head[..., 1:], null_val=np.nan)
                    speed_mse = masked_mse_np(gen_speed_heading[..., 0], gt_head[..., 1], null_val=np.nan)
                    direction_mse = masked_mse_np(gen_speed_heading[..., 1], gt_head[..., 2], null_val=np.nan)
                    print("The output mse is ", mse / 255 ** 2)
                    print("The speed mse is ", speed_mse / 255 ** 2)
                    print("The direction mse is ", direction_mse / 255 ** 2)

                    print("MSE on heading POS")
                    mse = masked_mse_np(gen_speed_heading, gt_head[..., 1:], null_val=0.0)
                    speed_mse = masked_mse_np(gen_speed_heading[..., 0], gt_head[..., 1], null_val=0.0)
                    direction_mse = masked_mse_np(gen_speed_heading[..., 1], gt_head[..., 2], null_val=0.0)
                    print("The output mse is ", mse / 255 ** 2)
                    print("The speed mse is ", speed_mse / 255 ** 2)
                    print("The direction mse is ", direction_mse / 255 ** 2)

            mavg_list.append(mavg_results)
            gt_list.append(gt)
            capsule_combine_list.append(capsule_combine)




    gt_list = np.stack(gt_list, axis=0)
    capsule_combine_list = np.stack(capsule_combine_list, axis=0)
    mavg_list = np.stack(mavg_list, axis=0)

    # print("img_gen shape is ", gx.shape)
    val_results_speed = np.sqrt(capsule_combine_list[..., 0] ** 2 + capsule_combine_list[..., 1] ** 2) * 255.0
    # print("val speed: ", val_results_speed, flush=True)
    val_results_heading = np.zeros_like(capsule_combine_list[..., 1])
    val_results_heading[(capsule_combine_list[..., 0] > 0) & (capsule_combine_list[..., 1] > 0)] = 85.0
    val_results_heading[(capsule_combine_list[..., 0] > 0) & (capsule_combine_list[..., 1] < 0)] = 255.0
    val_results_heading[(capsule_combine_list[..., 0] < 0) & (capsule_combine_list[..., 1] < 0)] = 170.0
    val_results_heading[(capsule_combine_list[..., 0] < 0) & (capsule_combine_list[..., 1] > 0)] = 1.0
    gen_speed_heading = np.stack([val_results_speed, val_results_heading], axis=-1)

    #
    # print("Evaluate on every pixels....")
    speed_threshold = mavg_list[..., 1] < 200

    gen_speed_heading[..., 1][speed_threshold] = mavg_list[..., 2][speed_threshold]
    mse = masked_mse_np(gen_speed_heading, gt_list[..., 1:], null_val=np.nan)
    speed_mse = masked_mse_np(gen_speed_heading[..., 0], gt_list[..., 1], null_val=np.nan)
    direction_mse = masked_mse_np(gen_speed_heading[..., 1], gt_list[..., 2], null_val=np.nan)
    print("The output mse is ", mse / 255**2)
    print("The speed mse is ", speed_mse / 255**2)
    print("The direction mse is ", direction_mse / 255**2)


    # print("Evaluate on valid pixels for Transformation...")
    print("Moving Avg is ")
    mse = masked_mse_np(mavg_list, gt_list, null_val=np.nan)
    speed_mse = masked_mse_np(mavg_list[..., 1], gt_list[..., 1], null_val=np.nan)
    direction_mse = masked_mse_np(mavg_list[..., 2], gt_list[..., 2], null_val=np.nan)
    print("The output mse is ", mse / 255**2)
    print("The speed mse is ", speed_mse / 255**2)
    print("The direction mse is ", direction_mse / 255**2)
    #
    # print("Evaluate on valid pixels for No Transformation...")
    # mse = masked_mse_np(pred_list_all, gt_list, null_val=0.0)
    # speed_mse = masked_mse_np(pred_list_all[..., 0], gt_list[..., 0], null_val=0.0)
    # direction_mse = masked_mse_np(pred_list_all[..., 1], gt_list[..., 1], null_val=0.0)
    # print("The output mse is ", mse)
    # print("The speed mse is ", speed_mse)
    # print("The direction mse is ", direction_mse)
    #
    # print("Evaluate on valid pixels for MAVG...")
    # # Evaluate on large gt speeds for direction
    # move_avg = np.stack(move_avg, axis=0)
    # mse = masked_mse_np(move_avg[..., 1:], gt_list, null_val=0.0)
    # speed_mse = masked_mse_np(move_avg[..., 1], gt_list[..., 0], null_val=0.0)
    # direction_mse = masked_mse_np(move_avg[..., 2], gt_list[..., 1], null_val=0.0)
    # print("The output mse is ", mse)
    # print("The speed mse is ", speed_mse)
    # print("The direction mse is ", direction_mse)
    #
    # large_gt_speed = move_avg[..., 1] > 0.5
    # direction_mse = masked_mse_np(pred_list[large_gt_speed, 1], gt_list[large_gt_speed, 1], null_val=np.nan)
    # print("The direction mse on large speed gt is ", direction_mse)

if __name__ == '__main__':
    tf.app.run()

