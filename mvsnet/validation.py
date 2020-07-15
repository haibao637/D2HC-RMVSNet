#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Training script.
"""

from __future__ import print_function

import os
import time
import sys
import math
import argparse
from random import randint
import visdom
import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
vis=visdom.Visdom(env="MVSNet")
sys.path.append("../")
from tools.common import Notify

from preprocess import *
from model import *
from loss import *
from homography_warping import get_homographies, homography_warping
from flowmodel import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# paths
tf.app.flags.DEFINE_string('dtu_data_root', '/home/haibao637/data/mvs_training/dtu/',
                           """Path to dtu dataset.""")
# tf.app.flags.DEFINE_string('log_dir', '/home/yanjianfeng/data5/tf_log',
#                            """Path to store the log.""")
# tf.app.flags.DEFINE_string('model_dir', '/home/yanjianfeng/data5/tf_model',
#                            """Path to save the model.""")
train_time = time.strftime("%y-%m-%d")
tf.app.flags.DEFINE_string('logdir', os.path.join('/home/haibao637/data/tf_log/', time.strftime("%y-%m-%d"),
                                                   time.strftime("%H:%M:%S")),
                           """Path to store the log.""")
tf.app.flags.DEFINE_string('model_dir', os.path.join('/home/haibao637/data/tf_models/'),
                           """Path to save the model.""")
tf.app.flags.DEFINE_boolean('train_dtu', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_boolean('use_pretrain', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_integer('ckpt_step',10000,
                            """ckpt step.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 3,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 128,
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 160,
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 128,
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25,
                          """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_float('interval_scale', 1.06,
                          """Downsample scale for building cost volume.""")

# network architectures
tf.app.flags.DEFINE_string('regularization', 'GRU',
                           """Regularization method.""")
tf.app.flags.DEFINE_boolean('refinement', False,
                            """Whether to apply depth map refinement for 1DCNNs""")

# training parameters
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Training batch size.""")
tf.app.flags.DEFINE_integer('epoch', 10,
                            """Training epoch number.""")
tf.app.flags.DEFINE_float('val_ratio', 0,
                          """Ratio of validation set when splitting dataset.""")
tf.app.flags.DEFINE_float('base_lr', 1e-3,
                          """Base learning rate.""")
tf.app.flags.DEFINE_integer('display', 1,
                            """Interval of loginfo display.""")
tf.app.flags.DEFINE_integer('stepvalue', 1250,
                            """Step interval to decay learning rate.""")
tf.app.flags.DEFINE_integer('snapshot', 5000,
                            """Step interval to save the model.""")
tf.app.flags.DEFINE_float('gamma', 0.9,
                          """Learning rate decay rate.""")
tf.app.flags.DEFINE_bool('inverse_depth', False,
                           """Whether to apply inverse depth for R-MVSNet""")
FLAGS = tf.app.flags.FLAGS


class MVSGenerator:
    """ data generator class, tf only accept generator without param """
    def __init__(self, sample_list, view_num,both=False):
        self.sample_list = sample_list
        random.shuffle(self.sample_list)
        self.view_num = view_num
        self.sample_num = len(sample_list)
        self.counter = 0
        self.both=both
    
    def __iter__(self):
        while True:
            for data in self.sample_list: 
                start_time = time.time()

                ###### read input data ######
                images = []
                cams = []
                depth_images=[]
                hists=[]
                
                # img = cv2.imread(data.image[0])
                for view in range(self.view_num):
                    img=cv2.imread(data[view][0])
                    img=cv2.resize(img,(FLAGS.max_w,FLAGS.max_h))
                    if view==0:
                        ref_image=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    img=center_image(img)
                    images.append(img)
                    cam=load_cam(open(data[view][1]), FLAGS.interval_scale)
                    # cam[1][3][1] = cam[1][3][1] * FLAGS.interval_scale
                    # cam=data[view].cam
                    cams.append(cam)
               
                    depth_start=cams[0][1,3,0] + cams[view][1,3,1]
                    # depth_start = cams[0][1, 3, 0] + cams[view][1,3,1]
                    # cams[view][1,3,1]*=2.0
                    depth_end = cams[0][1,3,0]+cams[view][1,3,1]*(FLAGS.max_d-2)
                    # cams[view][1,3,0]=depth_start
                    # cams[view][1,3,1]=(depth_end-depth_start)/(FLAGS.max_d-1)
                    depth_image = load_pfm(open(data[view][2]))
                    # depth_image=cv2.resize(depth_image,(depth_image.shape[1]*4,depth_image.shape[0]*4))
             
                    depth_image = mask_depth_image(depth_image, depth_start, depth_end)
                    # depth_image=np.load(data[view].depth)
                    # depth_image = mask_depth_image(depth_image, 0.5, 32)
                    depth_images.append(depth_image)
                # for view in range(self.view_num):
                #
                #     image = center_image(cv2.imread(data[2 * view]))
                #     cam = load_cam(open(data[2 * view + 1]))
                #     cam[1][3][1] = cam[1][3][1] * (FLAGS.interval_scale)
                #     images.append(image)
                #     cams.append(cam)
                #     depth_image = load_pfm(open(data[2 * self.view_num]))
                #     depth_images.append(depth_image)

                # mask out-of-range depth pixels (in a relaxed range)
                # depth_image = load_pfm(open(data.depths[0]))
                # depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                # depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]
                # depth_image = mask_depth_image(depth_image, 0, 10.0)
               
                
                
                # return mvs input
                self.counter += 1
                duration = time.time() - start_time
                images = np.stack(images, axis=0)
                cams = np.stack(cams, axis=0)
                depth_images=np.stack(depth_images,0)
                # depth_images=np.expand_dims(depth_images,-1)
                # print(depth_images.shape)
               
                # print('Forward pass: d_min = %f, d_max = %f.' % \
                #    (cams[0][1, 3, 0], cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]))
                yield (images, cams, depth_images,ref_image)

                # return backward mvs input for GRU
                # if FLAGS.regularization == 'GRU':
                #     self.counter += 1
                # start_time = time.time()
                


def valiation(traning_list):
    """ training mvsnet """
    training_sample_size = len(traning_list)
    # if FLAGS.regularization == 'GRU':
    training_sample_size = training_sample_size
    print('sample number: ', training_sample_size)
    with  tf.device('/cpu:0'):
        ########## data iterator #########
        # training generators
        training_generator = iter(MVSGenerator(traning_list, FLAGS.view_num))
        # generator_data_type = (tf.float32, tf.float32, tf.float32)
        generator_data_type = (tf.float32, tf.float32, tf.float32,tf.float32) 
        # dataset from generator
        training_set = tf.data.Dataset.from_generator(lambda: training_generator, generator_data_type)
        training_set = training_set.batch(FLAGS.batch_size)
        training_set = training_set.prefetch(buffer_size=1)
        # iterators
        training_iterator = training_set.make_initializable_iterator()
        ########## optimization options ##########
        global less_one_accuracy, less_three_accuracy, summaries, loss
        images, cams, depth_images,_ = training_iterator.get_next()
        images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
        cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
        depth_images.set_shape(tf.TensorShape([None,FLAGS.view_num, None, None, 1]))
        depth_start = tf.reshape(
            tf.slice(cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
        depth_interval = tf.reshape(
            tf.slice(cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
        depth_num = tf.cast(
            tf.reshape(tf.slice(cams, [0, 0, 1, 3, 2], [1, 1, 1, 1, 1]), []), "int32")
        depth_end = depth_start + (tf.cast(FLAGS.max_d, tf.float32) - 1) * depth_interval
                    
        depth_image = tf.squeeze(
            tf.slice(depth_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 1]), axis=1)
        with tf.device("/gpu:0"):
            init_depth_map, prob_map = inference_winner_take_all_4(images, cams, 
                FLAGS.max_d, depth_start, depth_end, reg_type='GRU', inverse_depth=FLAGS.inverse_depth)
            loss, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                        init_depth_map, depth_image, depth_interval)
        


        # initialization option
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        log=open('log_2.txt','a')
        with tf.Session(config=config) as sess:

            # initialization
            total_step = 0
            sess.run(init_op)
            # summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            for index in range(10000,600000,5000):
                # load pre-trained model
                pretrained_model_path = os.path.join("/home/haibao637/data/tf_models/","snetgn_singleHomo_uconv_lstm_128_both_32_fix_interval_scale_1.0","GRU", 'model.ckpt')
                restorer = tf.train.Saver(tf.global_variables())
                if os.path.exists('-'.join([pretrained_model_path, str(index)+".index"]))==False:
                    break
                restorer.restore(sess, '-'.join([pretrained_model_path, str(index)]))

                print(Notify.INFO, 'Pre-trained model restored from %s' %
                    ('-'.join([pretrained_model_path, str(index)])), Notify.ENDC)
                less_one_accuracies=[]
                less_three_accuracies=[]
                output_dir="/home/haibao637/validation"
                if os.path.exists(output_dir)==False:
                    os.makedirs(output_dir)
                for epoch in range(1):
                    # training of one epoch
                    step = 0
                    sess.run(training_iterator.initializer)
                    for idx in range(int(training_sample_size / FLAGS.num_gpus)):
                        # run one batch
                        start_time = time.time()
                        try:
                            out_less_one, out_less_three = sess.run(
                                [less_one_accuracy, less_three_accuracy] )
                            less_one_accuracies.append(out_less_one)
                            less_three_accuracies.append(out_less_three)
                        except tf.errors.OutOfRangeError:
                            print("End of dataset")  # ==> "End of dataset"
                            break
                        duration = time.time() - start_time
                        # np.save(os.path.join(output_dir,"%08d"%idx),out_images[:,0,...])
                        # np.save(os.path.join(output_dir, "%08d_d" % idx), out_depth_map)
                        # print info
                        if step % FLAGS.display == 0:
                            print(Notify.INFO,
                                'index, %d, step %d, (< 1px) = %.4f, (< 3px) = %.4f (%.3f sec/step)' %
                                (index, step, out_less_one, out_less_three, duration), Notify.ENDC)
                        # write summary
                        step += FLAGS.batch_size * FLAGS.num_gpus
                        total_step += FLAGS.batch_size * FLAGS.num_gpus

                print(Notify.INFO,
                    'validation  %d,  (< 1px) = %.4f, (< 3px) = %.4f' %
                    (index, sum(less_one_accuracies)/len(less_one_accuracies), sum(less_three_accuracies)/len(less_three_accuracies)), Notify.ENDC)
                log.writelines('%d,%.4f,%.4f\n'%(index, sum(less_one_accuracies)/len(less_one_accuracies), sum(less_three_accuracies)/len(less_three_accuracies)))
                acc1=sum(less_one_accuracies)/len(less_one_accuracies)
                acc3=sum(less_three_accuracies)/len(less_three_accuracies)
                vis.line(X=np.column_stack([index,index]),Y=np.column_stack([acc1,acc3]),
                win="mvsnet_accuracy",update='append',opts=dict(showlegend=True,legend=["accuracy_1","accuracy_3"]))

def main(argv=None):  # pylint: disable=unused-argument
    """ program entrance """
    # Prepare all training samples
    sample_list = gen_dtu_resized_path(FLAGS.dtu_data_root,mode='validation')
    # Shuffle
    # random.shuffle(sample_list)
    # Training entrance.
    # pretrained_model_path = os.path.join(FLAGS.model_dir, "19-07-22-8", FLAGS.regularization, 'model.ckpt')
    # print('-'.join([pretrained_model_path, str(5000)]))
    valiation(sample_list)


if __name__ == '__main__':
    print('Training MVSNet with %d views' % FLAGS.view_num)
    tf.app.run()
