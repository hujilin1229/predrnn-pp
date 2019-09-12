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
        if FLAGS.pretrained_model:
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

    FLAGS.save_dir += FLAGS.dataset_name + str(FLAGS.seq_length)
    FLAGS.pretrained_model = FLAGS.save_dir

    # FLAGS.save_dir += FLAGS.dataset_name
    # FLAGS.gen_frm_dir += FLAGS.dataset_name
    # if not tf.io.gfile.exists(FLAGS.save_dir):
    #     # tf.io.gfile.rmtree(FLAGS.save_dir)
    #     tf.io.gfile.makedirs(FLAGS.save_dir)
    # else:
    #     FLAGS.pretrained_model = FLAGS.save_dir
    # if not tf.io.gfile.exists(FLAGS.gen_frm_dir):
    #     # tf.io.gfile.rmtree(FLAGS.gen_frm_dir)
    #     tf.io.gfile.makedirs(FLAGS.gen_frm_dir)

    test_data_paths = os.path.join(FLAGS.valid_data_paths, FLAGS.dataset_name, FLAGS.dataset_name + '_validation')
    sub_files = preprocess.list_filenames(test_data_paths, [])

    output_path = './Results/predrnn/valiation/'
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

    se_total = 0.
    se_1 = 0.
    se_2 = 0.
    se_3 = 0.
    for f in sub_files:
        with h5py.File(os.path.join(test_data_paths, f), 'r') as h5_file:
            data = h5_file['array'][()]
            # get relevant training data pieces
            data = [data[y - FLAGS.input_length:y + FLAGS.seq_length - FLAGS.input_length] for y in indicies]
            data = np.stack(data, axis=0)
            # type casting

            test_dat = data.astype(np.float32) / 255.0
            test_dat = preprocess.reshape_patch(test_dat, FLAGS.patch_size_width, FLAGS.patch_size_height)
            batch_size = len(indicies)
            mask_true = np.zeros((batch_size,
                                  FLAGS.seq_length-FLAGS.input_length-1,
                                  FLAGS.img_height,
                                  FLAGS.img_width,
                                  FLAGS.patch_size_height*FLAGS.patch_size_width*FLAGS.img_channel))
            img_gen = model.test(test_dat, mask_true, batch_size)
            # concat outputs of different gpus along batch
            img_gen = np.concatenate(img_gen)
            img_gen = np.maximum(img_gen, 0)
            img_gen = np.minimum(img_gen, 1)
            img_gen = preprocess.reshape_patch_back(img_gen, FLAGS.patch_size_width, FLAGS.patch_size_height)
            img_gt = data[:, FLAGS.input_length:, ...].astype(np.float32) / 255.0
            se_total += np.sum((img_gt - img_gen)**2)

            se_1 += np.sum((img_gt[..., 0] - img_gen[..., 0]) ** 2)
            se_2 += np.sum((img_gt[..., 1] - img_gen[..., 1]) ** 2)
            se_3 += np.sum((img_gt[..., 2] - img_gen[..., 2]) ** 2)

            img_gen = np.uint8(img_gen*255)
            outfile = os.path.join(output_path, FLAGS.dataset_name, FLAGS.dataset_name + '_test', f)
            preprocess.write_data(img_gen, outfile)

    mse = se_total / (len(indicies) * len(sub_files) * 495 * 436 * 3 * 3)

    mse1 = se_1 / (len(indicies) * len(sub_files) * 495 * 436 * 3)
    mse2 = se_2 / (len(indicies) * len(sub_files) * 495 * 436 * 3)
    mse3 = se_3 / (len(indicies) * len(sub_files) * 495 * 436 * 3)
    print(FLAGS.dataset_name)
    print("MSE: ", mse)
    print("MSE_vol: ", mse1)
    print("MSE_sp: ", mse2)
    print("MSE_hd: ", mse3)

    print("Finished...")

if __name__ == '__main__':
    tf.app.run()

