# this file is adapted from train.py which is authored by 'yunbo'
__author__ = 'jilin'

import os.path
import numpy as np
import tensorflow as tf
from nets import models_factory
from data_provider import datasets_factory
from utils import preprocess
from utils import metrics

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
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/predrnn_pp_capsule_multitask',
                            'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_frm_dir', 'results/predrnn_pp_capsule_multitask',
                           'dir to store result.')
# model
tf.app.flags.DEFINE_string('model_name', 'predrnn_pp_capsule_multitask',
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
            output_list = models_factory.construct_multi_task_model(
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

    heading_dict = {1: 1, 2:85, 3: 170, 4: 255, 0:0}
    heading = FLAGS.heading
    FLAGS.save_dir += FLAGS.dataset_name + str(FLAGS.seq_length) + FLAGS.num_hidden + 'squash' + 'L1+L2+VALID' + 'multi-task'
    FLAGS.gen_frm_dir += FLAGS.dataset_name
    if not tf.io.gfile.exists(FLAGS.save_dir):
        # tf.io.gfile.rmtree(FLAGS.save_dir)
        tf.io.gfile.makedirs(FLAGS.save_dir)
    else:
        FLAGS.pretrained_model = FLAGS.save_dir
    if not tf.io.gfile.exists(FLAGS.gen_frm_dir):
        # tf.io.gfile.rmtree(FLAGS.gen_frm_dir)
        tf.io.gfile.makedirs(FLAGS.gen_frm_dir)

    train_data_paths = os.path.join(FLAGS.train_data_paths, FLAGS.dataset_name, FLAGS.dataset_name + '_training')
    valid_data_paths = os.path.join(FLAGS.valid_data_paths, FLAGS.dataset_name, FLAGS.dataset_name + '_validation')
    # load data
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        FLAGS.dataset_name, train_data_paths, valid_data_paths,
        FLAGS.batch_file, True, FLAGS.input_length, FLAGS.seq_length - FLAGS.input_length)

    # The following indicies are the start indicies of the 3 images to predict in the 288 time bins (0 to 287)
    # in each daily test file. These are time zone dependent. Berlin lies in UTC+2 whereas Istanbul and Moscow
    # lie in UTC+3.
    utcPlus2 = [30, 69, 126, 186, 234]
    utcPlus3 = [57, 114, 174, 222, 258]
    heading_table = np.array([[0, 0], [-1, 1], [1, 1], [-1, -1], [1, -1]], dtype=np.float32)

    indicies = utcPlus3
    if FLAGS.dataset_name == 'Berlin':
        indicies = utcPlus2

    # dims = train_input_handle.dims
    print("Initializing models", flush=True)
    model = Model()
    lr = FLAGS.lr

    delta = 0.00002
    base = 0.99998
    eta = 1

    for itr in range(1, FLAGS.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        imss = train_input_handle.get_batch()

        # # print("imss shape is ", imss.shape)
        tem_data = imss.copy()
        heading_image = imss[:, :, :, :, 2]*255
        heading_image = (heading_image // 85).astype(np.int8) + 1
        heading_image[tem_data[:, :, :, :, 2] == 0] = 0
        heading_image = heading_table[heading_image]

        speed_on_axis = np.expand_dims(imss[:, :, :, :, 1] / np.sqrt(2), axis=-1)
        imss = speed_on_axis * heading_image

        imss = preprocess.reshape_patch(imss, FLAGS.patch_size_width, FLAGS.patch_size_height)
        num_batches = imss.shape[0]
        for bi in range(0, num_batches, FLAGS.batch_size):
            ims = imss[bi:bi+FLAGS.batch_size]
            FLAGS.img_height = ims.shape[2]
            FLAGS.img_width = ims.shape[3]
            batch_size = ims.shape[0]
            if itr < 50000:
                eta -= delta
            else:
                eta = 0.0
            random_flip = np.random.random_sample(
                (batch_size, FLAGS.seq_length-FLAGS.input_length-1))
            true_token = (random_flip < eta)
            #true_token = (random_flip < pow(base,itr))
            ones = np.ones((FLAGS.img_height,
                            FLAGS.img_width,
                            int(FLAGS.patch_size_height*FLAGS.patch_size_width*FLAGS.img_channel)))
            zeros = np.zeros((int(FLAGS.img_height),
                              int(FLAGS.img_width),
                              int(FLAGS.patch_size_height*FLAGS.patch_size_width*FLAGS.img_channel)))
            mask_true = []
            for i in range(batch_size):
                for j in range(FLAGS.seq_length-FLAGS.input_length-1):
                    if true_token[i,j]:
                        mask_true.append(ones)
                    else:
                        mask_true.append(zeros)
            mask_true = np.array(mask_true)
            mask_true = np.reshape(mask_true, (batch_size,
                                               FLAGS.seq_length-FLAGS.input_length-1,
                                               int(FLAGS.img_height),
                                               int(FLAGS.img_width),
                                               int(FLAGS.patch_size_height*FLAGS.patch_size_width*FLAGS.img_channel)))
            cost, _ = model.train(ims, lr, mask_true, batch_size)

            if FLAGS.reverse_input:
                ims_rev = ims[:,::-1]
                cost2, _ = model.train(ims_rev, lr, mask_true, batch_size)
                cost = (cost + cost2) / 2

            if itr % FLAGS.display_interval == 0:
                print('itr: ' + str(itr), flush=True)
                print('training loss: ' + str(cost), flush=True)

        train_input_handle.next()
        if itr % FLAGS.test_interval == 0:
            print('test...', flush=True)
            epsilon = 0.2
            batch_size = len(indicies)
            test_input_handle.begin(do_shuffle = False)
            # res_path = os.path.join(FLAGS.gen_frm_dir, str(itr))
            # os.mkdir(res_path)
            avg_mse = 0
            batch_id = 0
            gt_list = []
            pred_list = []
            pred_list_all = []
            pred_vec = []
            move_avg = []
            img_mse, ssim, psnr, fmae, sharp= [],[],[],[],[]
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                img_mse.append(0)
                ssim.append(0)
                psnr.append(0)
                fmae.append(0)
                sharp.append(0)
            mask_true = np.zeros((batch_size,
                                  FLAGS.seq_length-FLAGS.input_length-1,
                                  FLAGS.img_height,
                                  FLAGS.img_width,
                                  FLAGS.patch_size_height*FLAGS.patch_size_width*FLAGS.img_channel))
            while(test_input_handle.no_batch_left() == False):
                batch_id = batch_id + 1
                test_ims = test_input_handle.get_test_batch(indicies)
                # get the ground truth
                gt_list.append(test_ims[:, FLAGS.input_length:, :, :, 1:])
                # cvt the heading to 0, 1, 2, 3, 4
                tem_data = test_ims.copy()
                heading_image = test_ims[:, :, :, :, 2] * 255
                heading_image = (heading_image // 85).astype(np.int8) + 1
                heading_image[tem_data[:, :, :, :, 2] == 0] = 0
                cvt_heading = heading_image.copy()

                # convert the data into speed vectors
                heading_selected = np.zeros_like(heading_image, np.int8)
                heading_selected[heading_image == heading] = heading
                heading_image = heading_selected
                heading_image = heading_table[heading_image]
                speed_on_axis = np.expand_dims(test_ims[:, :, :, :, 1] / np.sqrt(2), axis=-1)
                test_ims = speed_on_axis * heading_image

                # mavg filtered results
                mavg_results_all = cast_moving_avg(tem_data[:, :FLAGS.input_length, ...])
                mavg_results = np.zeros_like(mavg_results_all)
                # heading_image = np.expand_dims(heading_image, axis=-1)
                mavg_results[cvt_heading[:, FLAGS.input_length:, ...] == heading] = \
                    mavg_results_all[cvt_heading[:, FLAGS.input_length:, ...] == heading]
                move_avg.append(mavg_results)

                test_dat = preprocess.reshape_patch(test_ims, FLAGS.patch_size_width, FLAGS.patch_size_height)
                img_gen = model.test(test_dat, mask_true, batch_size)
                # concat outputs of different gpus along batch
                img_gen = np.concatenate(img_gen)
                # reshape the prediction has ndims=5
                img_gen = np.reshape(img_gen, (img_gen.shape[0], FLAGS.seq_length - FLAGS.input_length,
                                               FLAGS.img_height, FLAGS.img_width, -1))
                img_gen = preprocess.reshape_patch_back(img_gen, FLAGS.patch_size_width, FLAGS.patch_size_height)
                # print("Image Generates Shape is ", img_gen.shape)
                # MSE per frame

                img_gen_list = []
                img_gen_origin_list = []
                for i in range(FLAGS.seq_length - FLAGS.input_length):
                    x = tem_data[:,i + FLAGS.input_length,:,:, 1:]
                    gx = img_gen[:,i,:, :, :]

                    # print("img_gen shape is ", gx.shape)
                    val_results_speed = np.sqrt(gx[..., 0] ** 2 + gx[..., 1] ** 2)
                    # print("val speed: ", val_results_speed, flush=True)
                    val_results_heading = np.zeros_like(gx[..., 1])
                    val_results_heading[(gx[..., 0] > 0) & (gx[..., 1] > 0)] = 85.0 / 255.0
                    val_results_heading[(gx[..., 0] > 0) & (gx[..., 1] < 0)] = 255.0 / 255.0
                    val_results_heading[(gx[..., 0] < 0) & (gx[..., 1] < 0)] = 170.0 / 255.0
                    val_results_heading[(gx[..., 0] < 0) & (gx[..., 1] > 0)] = 1.0 / 255.0

                    gen_speed_heading = np.stack([val_results_speed, val_results_heading], axis=-1)
                    img_gen_origin_list.append(gen_speed_heading)

                    # Transformation according to moving average direction when mavg speed is small
                    val_results_heading[mavg_results[:, i, :, :, 1] < epsilon] = \
                        mavg_results[:, i, :, :, 2][mavg_results[:, i, :, :, 1] < epsilon]
                    gx = np.stack([val_results_speed, val_results_heading], axis=-1)
                    img_gen_list.append(gx)

                    fmae[i] += metrics.batch_mae_frame_float(gx, x)
                    gx = np.maximum(gx, 0)
                    gx = np.minimum(gx, 1)
                    mse = np.square(x - gx).sum()
                    img_mse[i] += mse
                    avg_mse += mse

                img_gen_list = np.stack(img_gen_list, axis=1)
                img_gen_origin_list = np.stack(img_gen_origin_list, axis=1)
                pred_list_all.append(img_gen_origin_list)
                pred_list.append(img_gen_list)
                pred_vec.append(img_gen)
                test_input_handle.next()

            avg_mse = avg_mse / (batch_id*batch_size*FLAGS.img_height *
                                 FLAGS.img_width * FLAGS.patch_size_height *
                                 FLAGS.patch_size_width * FLAGS.img_channel * len(img_mse))
            print('mse per seq: ' + str(avg_mse), flush=True)
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(img_mse[i] / (batch_id*batch_size*FLAGS.img_height *
                                 FLAGS.img_width * FLAGS.patch_size_height *
                                 FLAGS.patch_size_width * FLAGS.img_channel))

            gt_list_all = np.stack(gt_list, axis=0)
            # GT filtered to the direction required
            gt_list = np.zeros_like(gt_list_all)
            gt_list[gt_list_all[..., 1]*255 == heading_dict[heading]] = \
                gt_list_all[gt_list_all[..., 1]*255 == heading_dict[heading]]

            pred_list = np.stack(pred_list, axis=0)
            pred_list_all = np.stack(pred_list_all, axis=0)

            print("Evaluate on every pixels....")
            mse = masked_mse_np(pred_list, gt_list, null_val=np.nan)
            speed_mse = masked_mse_np(pred_list[..., 0], gt_list[..., 0], null_val=np.nan)
            direction_mse = masked_mse_np(pred_list[..., 1], gt_list[..., 1], null_val=np.nan)
            print("The output mse is ", mse)
            print("The speed mse is ", speed_mse)
            print("The direction mse is ", direction_mse)

            print("Evaluate on valid pixels for Transformation...")
            mse = masked_mse_np(pred_list, gt_list, null_val=0.0)
            speed_mse = masked_mse_np(pred_list[..., 0], gt_list[..., 0], null_val=0.0)
            direction_mse = masked_mse_np(pred_list[..., 1], gt_list[..., 1], null_val=0.0)
            print("The output mse is ", mse)
            print("The speed mse is ", speed_mse)
            print("The direction mse is ", direction_mse)

            print("Evaluate on valid pixels for No Transformation...")
            mse = masked_mse_np(pred_list_all, gt_list, null_val=0.0)
            speed_mse = masked_mse_np(pred_list_all[..., 0], gt_list[..., 0], null_val=0.0)
            direction_mse = masked_mse_np(pred_list_all[..., 1], gt_list[..., 1], null_val=0.0)
            print("The output mse is ", mse)
            print("The speed mse is ", speed_mse)
            print("The direction mse is ", direction_mse)

            print("Evaluate on valid pixels for MAVG...")
            # Evaluate on large gt speeds for direction
            move_avg = np.stack(move_avg, axis=0)
            mse = masked_mse_np(move_avg[..., 1:], gt_list, null_val=0.0)
            speed_mse = masked_mse_np(move_avg[..., 1], gt_list[..., 0], null_val=0.0)
            direction_mse = masked_mse_np(move_avg[..., 2], gt_list[..., 1], null_val=0.0)
            print("The output mse is ", mse)
            print("The speed mse is ", speed_mse)
            print("The direction mse is ", direction_mse)

            large_gt_speed = move_avg[..., 1] >= epsilon
            move_avg[..., 2][large_gt_speed] = pred_list_all[large_gt_speed, 1]
            direction_mse = masked_mse_np(move_avg[..., 2], gt_list[..., 1], null_val=0.0)
            print(f"The direction of combined mavg and large speed~({epsilon}) prediction is ", direction_mse)

            direction_mse = masked_mse_np(pred_list_all[large_gt_speed, 1], gt_list[large_gt_speed, 1], null_val=0.0)
            print("The direction mse on large speed gt is ", direction_mse)


        if itr % FLAGS.snapshot_interval == 0:
            model.save(itr)

if __name__ == '__main__':
    tf.app.run()

