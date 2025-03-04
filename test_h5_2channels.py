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
        t = np.mean(data_slice, axis=1)

        # # sol 4. consider future average: performance little lift: mse: 3059.7886
        data_slice_future = []
        for j in range(4):
            data_slice_future.append(np.concatenate([data_slice[j, ...], data_slice[j + 1, ...]], axis=0))
        data_slice_future.append(np.concatenate([data_slice[4, ...], data_slice[4, ...]], axis=0))
        data_slice_future = np.stack(data_slice_future, axis=0)
        t = np.mean(data_slice_future, axis=1)

        # Return the normal operation
        t = np.expand_dims(t, axis=1)
        prediction.append(t)
        data = np.concatenate([data, t], axis=1)

    prediction = np.concatenate(prediction, axis=1)
    prediction = np.around(prediction)
    prediction = prediction.astype(np.uint8)

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
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/predrnn_pp',
                            'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_frm_dir', 'results/predrnn_pp',
                           'dir to store result.')
# model
tf.app.flags.DEFINE_string('model_name', 'predrnn_pp',
                           'The name of the architecture.')
tf.app.flags.DEFINE_string('pretrained_model', '',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_string('best_model', '',
                           'file of the best model to initialize from.')
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
tf.app.flags.DEFINE_integer('img_channel', 3,
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

tf.app.flags.DEFINE_boolean('layer_norm', True,
                            'whether to apply tensor layer norm.')
# optimization
tf.app.flags.DEFINE_float('lr', 0.001,
                          'base learning rate.')
tf.app.flags.DEFINE_boolean('reverse_input', True,
                            'whether to reverse the input frames while training.')
tf.app.flags.DEFINE_integer('batch_file', 1,
                            'num of file per batch for training.')
tf.app.flags.DEFINE_integer('max_iterations', 1000,
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
        loss_train = []
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
        # if FLAGS.pretrained_model:
        #     print("pretrained_model dir: ", FLAGS.pretrained_model)
        #     print("latest checkpoint: ", tf.train.latest_checkpoint(FLAGS.pretrained_model))
        #     self.saver.restore(self.sess, tf.train.latest_checkpoint(FLAGS.pretrained_model))
        if FLAGS.best_model:
            try:
                print("the best model dir: ", FLAGS.best_model)
                self.saver.restore(self.sess, FLAGS.best_model)
            except:
                print("pretrained_model dir: ", FLAGS.pretrained_model)
                print("latest checkpoint: ", tf.train.latest_checkpoint(FLAGS.pretrained_model))
                self.saver.restore(self.sess, tf.train.latest_checkpoint(FLAGS.pretrained_model))

    def train(self, inputs, lr, mask_true, batch_size):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.tf_lr: lr})
        feed_dict.update({self.mask_true: mask_true})
        feed_dict.update({self.batchsize: batch_size})
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
        return loss

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

    # FLAGS.save_dir += FLAGS.dataset_name
    # FLAGS.gen_frm_dir += FLAGS.dataset_name
    # if tf.io.gfile.exists(FLAGS.save_dir):
    #     tf.io.gfile.rmtree(FLAGS.save_dir)
    # tf.io.gfile.makedirs(FLAGS.save_dir)
    # if tf.io.gfile.exists(FLAGS.gen_frm_dir):
    #     tf.io.gfile.rmtree(FLAGS.gen_frm_dir)
    # tf.io.gfile.makedirs(FLAGS.gen_frm_dir)

    FLAGS.save_dir += FLAGS.dataset_name + str(FLAGS.seq_length) + FLAGS.num_hidden
    print(FLAGS.save_dir)
    # FLAGS.best_model = FLAGS.save_dir + '/best.ckpt'
    FLAGS.best_model = FLAGS.save_dir + f'/best_channels{FLAGS.img_channel}.ckpt'
    # FLAGS.best_model = FLAGS.save_dir + f'/best_channels{FLAGS.img_channel}_weighted.ckpt'
    # FLAGS.save_dir += FLAGS.dataset_name
    FLAGS.pretrained_model = FLAGS.save_dir

    process_data_dir = os.path.join(FLAGS.valid_data_paths, FLAGS.dataset_name, 'process_0.5')
    node_pos_file_2in1 = os.path.join(process_data_dir, 'node_pos_0.5.npy')
    node_pos = np.load(node_pos_file_2in1)

    test_data_paths = os.path.join(FLAGS.valid_data_paths, FLAGS.dataset_name, FLAGS.dataset_name + '_' + FLAGS.mode)
    sub_files = preprocess.list_filenames(test_data_paths, [])

    output_path = f'./Results/predrnn/t{FLAGS.test_time}_{FLAGS.mode}/'
    # output_path = f'./Results/predrnn/t14/'
    preprocess.create_directory_structure(output_path)
    # The following indicies are the start indicies of the 3 images to predict in the 288 time bins (0 to 287)
    # in each daily test file. These are time zone dependent. Berlin lies in UTC+2 whereas Istanbul and Moscow
    # lie in UTC+3.
    utcPlus2 = [30, 69, 126, 186, 234]
    utcPlus3 = [57, 114, 174, 222, 258]
    indicies = utcPlus3
    if FLAGS.dataset_name == 'Berlin':
        indicies = utcPlus2

    print("Initializing models", flush=True)
    model = Model()

    step = 6
    se_total = 0.
    se_1 = 0.
    se_2 = 0.
    se_3 = 0.
    gt_list = []
    pred_list = []
    mavg_list = []
    for f in sub_files:
        with h5py.File(os.path.join(test_data_paths, f), 'r') as h5_file:
            data = h5_file['array'][()]
            # Query the Moving Average Data
            prev_data = [data[y - step:y] for y in indicies]
            prev_data = np.stack(prev_data, axis=0)
            # type casting
            # prev_data = prev_data.astype(np.float32) / 255.0
            # mavg_pred = cast_moving_avg(prev_data)
            # mavg_list.append(mavg_pred)

            # get relevant training data pieces
            data = [data[y - FLAGS.input_length:y + FLAGS.seq_length - FLAGS.input_length] for y in indicies]
            data = np.stack(data, axis=0)
            # select the data channel as wished
            data = data[..., :FLAGS.img_channel]

            # all validation data is applied
            # data = np.reshape(data,(-1, FLAGS.seq_length,
            #                     FLAGS.img_height*FLAGS.patch_size_height, FLAGS.img_width*FLAGS.patch_size_width, 3))
            # type casting
            test_dat = data.astype(np.float32) / 255.0
            test_dat = preprocess.reshape_patch(test_dat, FLAGS.patch_size_width, FLAGS.patch_size_height)
            batch_size = data.shape[0]
            mask_true = np.zeros((batch_size,
                                  FLAGS.seq_length-FLAGS.input_length-1,
                                  FLAGS.img_height,
                                  FLAGS.img_width,
                                  FLAGS.patch_size_height*FLAGS.patch_size_width*FLAGS.img_channel))
            img_gen = model.test(test_dat, mask_true, batch_size)
            # concat outputs of different gpus along batch
            # img_gen = np.concatenate(img_gen)
            img_gen = img_gen[0]
            img_gen = np.maximum(img_gen, 0)
            img_gen = np.minimum(img_gen, 1)
            img_gen = preprocess.reshape_patch_back(img_gen, FLAGS.patch_size_width, FLAGS.patch_size_height)
            img_gt = data[:, FLAGS.input_length:, ...].astype(np.float32) / 255.0

            gt_list.append(img_gt)
            pred_list.append(img_gen)
            se_total += np.sum((img_gt - img_gen)**2)

            se_1 += np.sum((img_gt[..., 0] - img_gen[..., 0]) ** 2)
            se_2 += np.sum((img_gt[..., 1] - img_gen[..., 1]) ** 2)
            # se_3 += np.sum((img_gt[..., 2] - img_gen[..., 2]) ** 2)

            img_gen = np.uint8(img_gen*255)
            outfile = os.path.join(output_path, FLAGS.dataset_name, FLAGS.dataset_name + '_test', f)
            preprocess.write_data(img_gen, outfile)


    # mse = se_total / (len(indicies) * len(sub_files) * 495 * 436 * 3 * 3)
    #
    # mse1 = se_1 / (len(indicies) * len(sub_files) * 495 * 436 * 3)
    # mse2 = se_2 / (len(indicies) * len(sub_files) * 495 * 436 * 3)
    # # mse3 = se_3 / (len(indicies) * len(sub_files) * 495 * 436 * 3)
    # print(FLAGS.dataset_name)
    # print("MSE: ", mse)
    # print("MSE_vol: ", mse1)
    # print("MSE_sp: ", mse2)
    # # print("MSE_hd: ", mse3)
    #
    # pred_list = np.stack(pred_list, axis=0)
    # gt_list = np.stack(gt_list, axis=0)
    # mavg_list = np.stack(mavg_list, axis=0)
    #
    # array_mse = masked_mse_np(mavg_list, gt_list, np.nan)
    # print(f'MAVG {step} MSE: ', array_mse)
    #
    # # adapt pred on non_zero mavg pred only
    # pred_list_copy = np.zeros_like(pred_list)
    # pred_list_copy[mavg_list > 0] = pred_list[mavg_list > 0]
    #
    # array_mse = masked_mse_np(pred_list_copy, gt_list, np.nan)
    # print(f'PRED+MAVG {step} MSE: ', array_mse)
    #
    # # Evaluate on nodes
    # # Check MSE on node_pos
    # img_gt_node = gt_list[:, :, :, node_pos[:, 0], node_pos[:, 1], :].astype(np.float32)
    # img_gen_node = pred_list[:, :, :, node_pos[:, 0], node_pos[:, 1], :].astype(np.float32)
    # mse_node_all = masked_mse_np(img_gen_node, img_gt_node, np.nan)
    # mse_node_volume = masked_mse_np(img_gen_node[..., 0], img_gt_node[..., 0], np.nan)
    # mse_node_speed = masked_mse_np(img_gen_node[..., 1], img_gt_node[..., 1], np.nan)
    # mse_node_direction = masked_mse_np(img_gen_node[..., 2], img_gt_node[..., 2], np.nan)
    # print("Results on Node Pos: ")
    # print("MSE: ", mse_node_all)
    # print("Volume mse: ", mse_node_volume)
    # print("Speed mse: ", mse_node_speed)
    # print("Direction mse: ", mse_node_direction)
    #
    # print("Evaluating on Condensed Graph....")
    # seq_length = np.shape(gt_list)[2]
    # img_height = np.shape(gt_list)[3]
    # img_width = np.shape(gt_list)[4]
    # num_channels = np.shape(gt_list)[5]
    # gt_list = np.reshape(gt_list, [-1, seq_length,
    #                             int(img_height / FLAGS.patch_size_height), FLAGS.patch_size_height,
    #                             int(img_width / FLAGS.patch_size_width), FLAGS.patch_size_width,
    #                             num_channels])
    # gt_list = np.transpose(gt_list, [0, 1, 2, 4, 3, 5, 6])
    #
    # pred_list = np.reshape(pred_list, [-1, seq_length,
    #                                int(img_height / FLAGS.patch_size_height), FLAGS.patch_size_height,
    #                                int(img_width / FLAGS.patch_size_width), FLAGS.patch_size_width,
    #                                num_channels])
    # pred_list = np.transpose(pred_list, [0, 1, 2, 4, 3, 5, 6])
    #
    # node_pos = preprocess.construct_road_network_from_grid_condense(FLAGS.patch_size_height, FLAGS.patch_size_width,
    #                                                                 test_data_paths)
    #
    # img_gt_node = gt_list[:, :, node_pos[:, 0], node_pos[:, 1], ...].astype(np.float32)
    # img_gen_node = pred_list[:, :, node_pos[:, 0], node_pos[:, 1], ...].astype(np.float32)
    # mse_node_all = masked_mse_np(img_gen_node, img_gt_node, np.nan)
    # mse_node_volume = masked_mse_np(img_gen_node[..., 0], img_gt_node[..., 0], np.nan)
    # mse_node_speed = masked_mse_np(img_gen_node[..., 1], img_gt_node[..., 1], np.nan)
    # mse_node_direction = masked_mse_np(img_gen_node[..., 2], img_gt_node[..., 2], np.nan)
    # print("MSE: ", mse_node_all)
    # print("Volume mse: ", mse_node_volume)
    # print("Speed mse: ", mse_node_speed)
    # print("Direction mse: ", mse_node_direction)

    print("Finished...")



if __name__ == '__main__':
    tf.app.run()

