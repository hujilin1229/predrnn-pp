__author__ = 'yunbo'

import os.path
import time
import numpy as np
import tensorflow as tf
import cv2
import sys
import random
from nets import models_factory
from data_provider import datasets_factory
from utils import preprocess
from utils import metrics
from utils.preprocess import list_filenames
# from skimage.measure import compare_ssim

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
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/mnist_predrnn_pp',
                            'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_frm_dir', 'results/mnist_predrnn_pp',
                           'dir to store result.')
tf.app.flags.DEFINE_integer('down_sample', 4,
                            'ratio to down sample.')
# model
tf.app.flags.DEFINE_string('model_name', 'predrnn_pp',
                           'The name of the architecture.')
tf.app.flags.DEFINE_string('pretrained_model', '',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('input_length', 12,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('seq_length', 15,
                            'total input and output length.')
tf.app.flags.DEFINE_integer('img_width', 64,
                            'input image width.')
tf.app.flags.DEFINE_integer('img_height', 64,
                            'input image width.')
tf.app.flags.DEFINE_integer('img_channel', 1,
                            'number of image channel.')
tf.app.flags.DEFINE_integer('stride', 1,
                            'stride of a convlstm layer.')
tf.app.flags.DEFINE_integer('filter_size', 5,
                            'filter of a convlstm layer.')
tf.app.flags.DEFINE_string('num_hidden', '32,32',
                           'COMMA separated number of units in a convlstm layer.')
tf.app.flags.DEFINE_integer('patch_size', 4,
                            'patch size on one dimension.')
tf.app.flags.DEFINE_boolean('layer_norm', True,
                            'whether to apply tensor layer norm.')
# optimization
tf.app.flags.DEFINE_float('lr', 0.001,
                          'base learning rate.')
tf.app.flags.DEFINE_boolean('reverse_input', True,
                            'whether to reverse the input frames while training.')
tf.app.flags.DEFINE_integer('batch_size', 8,
                            'batch size for training.')
tf.app.flags.DEFINE_integer('max_iterations', 80000,
                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 1,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 200,
                            'number of iters for test.')
tf.app.flags.DEFINE_integer('snapshot_interval', 10000,
                            'number of iters saving models.')

class Model(object):
    def __init__(self):
        # inputs
        self.x = tf.placeholder(tf.float32,
                                [FLAGS.batch_size,
                                 FLAGS.seq_length,
                                 FLAGS.img_height,
                                 FLAGS.img_width,
                                 int(FLAGS.patch_size*FLAGS.patch_size*FLAGS.img_channel)])

        self.mask_true = tf.placeholder(tf.float32,
                                        [FLAGS.batch_size,
                                         FLAGS.seq_length-FLAGS.input_length-1,
                                         FLAGS.img_height,
                                         FLAGS.img_width,
                                         int(FLAGS.patch_size*FLAGS.patch_size*FLAGS.img_channel)])

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
                FLAGS.layer_norm)
            gen_ims = output_list[0]
            loss = output_list[1]
            pred_ims = gen_ims[:,FLAGS.input_length-1:]
            self.loss_train = loss / FLAGS.batch_size
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
            self.saver.restore(self.sess, FLAGS.pretrained_model)

    def train(self, inputs, lr, mask_true):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.tf_lr: lr})
        feed_dict.update({self.mask_true: mask_true})
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
        return loss

    def test(self, inputs, mask_true):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.mask_true: mask_true})
        gen_ims = self.sess.run(self.pred_seq, feed_dict)
        return gen_ims

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir, flush=True)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.save_dir):
        tf.gfile.DeleteRecursively(FLAGS.save_dir)
    tf.gfile.MakeDirs(FLAGS.save_dir)
    if tf.gfile.Exists(FLAGS.gen_frm_dir):
        tf.gfile.DeleteRecursively(FLAGS.gen_frm_dir)
    tf.gfile.MakeDirs(FLAGS.gen_frm_dir)

    train_data_paths = os.path.join(FLAGS.train_data_paths, FLAGS.dataset_name,
                                    'train_speed_down_sample{}.npz'.format(FLAGS.down_sample))
    valid_data_paths = os.path.join(FLAGS.valid_data_paths, FLAGS.dataset_name,
                                    'valid_speed_down_sample{}.npz'.format(FLAGS.down_sample))
    # load data
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        FLAGS.dataset_name, train_data_paths, valid_data_paths,
        FLAGS.batch_size, True, FLAGS.down_sample, FLAGS.input_length, FLAGS.seq_length - FLAGS.input_length)

    cities = ['Berlin', 'Istanbul', 'Moscow']
    # The following indicies are the start indicies of the 3 images to predict in the 288 time bins (0 to 287)
    # in each daily test file. These are time zone dependent. Berlin lies in UTC+2 whereas Istanbul and Moscow
    # lie in UTC+3.
    utcPlus2 = [30, 69, 126, 186, 234]
    utcPlus3 = [57, 114, 174, 222, 258]
    indicies = utcPlus3
    if FLAGS.dataset_name == 'Berlin':
        indicies = utcPlus2

    dims = train_input_handle.dims
    FLAGS.img_height = dims[-2]
    FLAGS.img_width = dims[-1]
    print("Initializing models", flush=True)
    model = Model()
    lr = FLAGS.lr

    delta = 0.00002
    base = 0.99998
    eta = 1

    for itr in range(1, FLAGS.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims = train_input_handle.get_batch()
        ims = preprocess.reshape_patch(ims, FLAGS.patch_size)

        if itr < 50000:
            eta -= delta
        else:
            eta = 0.0
        random_flip = np.random.random_sample(
            (FLAGS.batch_size, FLAGS.seq_length-FLAGS.input_length-1))
        true_token = (random_flip < eta)
        #true_token = (random_flip < pow(base,itr))
        ones = np.ones((FLAGS.img_height,
                        FLAGS.img_width,
                        int(FLAGS.patch_size**2*FLAGS.img_channel)))
        zeros = np.zeros((int(FLAGS.img_height),
                          int(FLAGS.img_width),
                          int(FLAGS.patch_size**2*FLAGS.img_channel)))
        mask_true = []
        for i in range(FLAGS.batch_size):
            for j in range(FLAGS.seq_length-FLAGS.input_length-1):
                if true_token[i,j]:
                    mask_true.append(ones)
                else:
                    mask_true.append(zeros)
        mask_true = np.array(mask_true)
        mask_true = np.reshape(mask_true, (FLAGS.batch_size,
                                           FLAGS.seq_length-FLAGS.input_length-1,
                                           int(FLAGS.img_height),
                                           int(FLAGS.img_width),
                                           int(FLAGS.patch_size**2*FLAGS.img_channel)))
        cost = model.train(ims, lr, mask_true)
        if FLAGS.reverse_input:
            ims_rev = ims[:,::-1]
            cost += model.train(ims_rev, lr, mask_true)
            cost = cost/2

        if itr % FLAGS.display_interval == 0:
            print('itr: ' + str(itr), flush=True)
            print('training loss: ' + str(cost), flush=True)

        if itr % FLAGS.test_interval == 0:
            print('test...', flush=True)
            test_input_handle.begin(do_shuffle = False)
            res_path = os.path.join(FLAGS.gen_frm_dir, str(itr))
            os.mkdir(res_path)
            avg_mse = 0
            batch_id = 0
            img_mse, ssim, psnr, fmae, sharp= [],[],[],[],[]
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                img_mse.append(0)
                ssim.append(0)
                psnr.append(0)
                fmae.append(0)
                sharp.append(0)
            mask_true = np.zeros((FLAGS.batch_size,
                                  FLAGS.seq_length-FLAGS.input_length-1,
                                  FLAGS.img_height,
                                  FLAGS.img_width,
                                  FLAGS.patch_size**2*FLAGS.img_channel))
            while(test_input_handle.no_batch_left() == False):
                batch_id = batch_id + 1
                test_ims = test_input_handle.get_batch()
                test_dat = preprocess.reshape_patch(test_ims, FLAGS.patch_size)
                img_gen = model.test(test_dat, mask_true)

                # concat outputs of different gpus along batch
                img_gen = np.concatenate(img_gen)
                img_gen = preprocess.reshape_patch_back(img_gen, FLAGS.patch_size)
                # MSE per frame
                for i in range(FLAGS.seq_length - FLAGS.input_length):
                    x = test_ims[:,i + FLAGS.input_length,:,:,0]
                    gx = img_gen[:,i,:,:,0]
                    fmae[i] += metrics.batch_mae_frame_float(gx, x)
                    gx = np.maximum(gx, 0)
                    gx = np.minimum(gx, 1)
                    mse = np.square(x - gx).sum()
                    img_mse[i] += mse
                    avg_mse += mse

                    real_frm = np.uint8(x * 255)
                    pred_frm = np.uint8(gx * 255)
                    psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
                    for b in range(FLAGS.batch_size):
                        sharp[i] += np.max(
                            cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b],3)))
                        # score, _ = compare_ssim(pred_frm[b],real_frm[b],full=True)
                        # ssim[i] += score

                # save prediction examples
                if batch_id <= 10:
                    path = os.path.join(res_path, str(batch_id))
                    os.mkdir(path)
                    for i in range(FLAGS.seq_length):
                        name = 'gt' + str(i+1) + '.png'
                        file_name = os.path.join(path, name)
                        img_gt = np.uint8(test_ims[0,i,:,:,:] * 255)
                        cv2.imwrite(file_name, img_gt)
                    for i in range(FLAGS.seq_length-FLAGS.input_length):
                        name = 'pd' + str(i+1+FLAGS.input_length) + '.png'
                        file_name = os.path.join(path, name)
                        img_pd = img_gen[0,i,:,:,:]
                        img_pd = np.maximum(img_pd, 0)
                        img_pd = np.minimum(img_pd, 1)
                        img_pd = np.uint8(img_pd * 255)
                        cv2.imwrite(file_name, img_pd)
                test_input_handle.next()
            avg_mse = avg_mse / (batch_id*FLAGS.batch_size)
            print('mse per seq: ' + str(avg_mse), flush=True)
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(img_mse[i] / (batch_id*FLAGS.batch_size))
            psnr = np.asarray(psnr, dtype=np.float32)/batch_id
            fmae = np.asarray(fmae, dtype=np.float32)/batch_id
            sharp = np.asarray(sharp, dtype=np.float32)/(FLAGS.batch_size*batch_id)
            print('psnr per frame: ' + str(np.mean(psnr)), flush=True)
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(psnr[i], flush=True)
            print('fmae per frame: ' + str(np.mean(fmae)))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(fmae[i], flush=True)
            print('sharpness per frame: ' + str(np.mean(sharp)))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(sharp[i], flush=True)

            # test with file
            valid_data_path = os.path.join(FLAGS.train_data_paths, FLAGS.dataset_name,'{}_validation'.format(FLAGS.dataset_name))
            files = list_filenames(valid_data_path)
            output_all = []
            labels_all = []
            for f in files:
                valid_file = valid_data_path + '/' + f
                valid_input, raw_output = datasets_factory.test_validation_provider(
                    valid_file, indicies, down_sample=FLAGS.down_sample, seq_len=FLAGS.input_length,
                    horizon=FLAGS.seq_length - FLAGS.input_length)
                valid_input = valid_input.astype(np.float) / 255.0
                labels_all.append(raw_output)
                num_tests = len(indicies)
                num_partitions = int(np.ceil(num_tests / FLAGS.batch_size))
                for i in range(num_partitions):
                    valid_input_i = valid_input[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                    num_input_i = valid_input_i.shape[0]
                    if num_input_i < FLAGS.batch_size:
                        zeros_fill_in = np.zeros((FLAGS.batch_size - num_input_i,
                                                  FLAGS.seq_length,
                                                  FLAGS.img_height,
                                                  FLAGS.img_width,
                                                  FLAGS.img_channel))
                        valid_input_i = np.concatenate([valid_input_i, zeros_fill_in], axis=0)
                    img_gen = model.test(valid_input_i, mask_true)
                    output_all.append(img_gen[0][:num_input_i])

            output_all = np.concatenate(output_all, axis=0)
            labels_all = np.concatenate(labels_all, axis=0)
            origin_height = labels_all.shape[-2]
            origin_width = labels_all.shape[-3]
            output_resize = []
            for i in range(output_all.shape[0]):
                output_i = []
                for j in range(output_all.shape[1]):
                    tmp_data = output_all[i, j, 1, :, :]
                    tmp_data = cv2.resize(tmp_data, (origin_height, origin_width))
                    tmp_data = np.expand_dims(tmp_data, axis=0)
                    output_i.append(tmp_data)
                output_i = np.stack(output_i, axis=0)
                output_resize.append(output_i)
            output_resize = np.stack(output_resize, axis=0)

            output_resize *= 255.0
            labels_all = np.expand_dims(labels_all[..., 1], axis=2)
            valid_mse = masked_mse_np(output_resize, labels_all, np.nan)

            print("validation mse is ", valid_mse, flush=True)

        if itr % FLAGS.snapshot_interval == 0:
            model.save(itr)

        train_input_handle.next()

if __name__ == '__main__':
    tf.app.run()

