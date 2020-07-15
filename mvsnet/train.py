#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
# from valid_disp import validation
# import high_dim_filter_loader

sys.path.append("../")
from tools.common import Notify

from preprocess import *
from model import *
from loss import *
from homography_warping import get_homographies, homography_warping
# custom_module = high_dim_filter_loader.custom_module
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# paths

tf.app.flags.DEFINE_string('dtu_data_root', '/home/haibao637/data/mvs_training/dtu/',
                           """Path to dtu dataset.""")

tf.app.flags.DEFINE_string('demon_data_root', '/home/haibao637/data/DPSNet/dataset/train/',
                           """Path to dtu dataset.""")
# tf.app.flags.DEFINE_string('eth3d_data_root', '/home/yanjianfeng/data6//eth3d_training/eth3d/',
#                            """Path to dtu dataset.""")
# tf.app.flags.DEFINE_string('logdirs', '/home/yanjianfeng/data5/tf_log',
#                            """Path to store the log.""")
# tf.app.flags.DEFINE_string('model_dir', '/home/yanjianfeng/data5/tf_model',
#                            """Path to save the model.""")
train_time=time.strftime("%y-%m-%d-10")
model_version="snetgn_singlhomo_ulstm_128_both_interval_scale_1.08_lstmgate"
tf.app.flags.DEFINE_string('logdirs', os.path.join('/home/haibao637/data/tf_log/',model_version,time.strftime("%H:%M:%S")),
						   """Path to store the log.""")
tf.app.flags.DEFINE_string('model_dir', os.path.join('/home/haibao637/data/tf_models/') ,
						   """Path to save the model.""")
tf.app.flags.DEFINE_boolean('train_dtu', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_boolean('use_pretrain',False,
                            """Whether to train.""")
tf.app.flags.DEFINE_integer('ckpt_step',100000,
                            """ckpt step.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num',3,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 128,
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 160,
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 128,
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('interval_scale', 1.08,
                            """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_float('disp_size', 10.0,
                            """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_bool('hist', False,
                            """Downsample scale for building cost volume.""")
# network architectures
tf.app.flags.DEFINE_string('regularization', 'GRU',
                           """Regularization method.""")
tf.app.flags.DEFINE_boolean('refinement', False,
                           """Whether to apply depth map refinement for 1DCNNs""")

# training parameters
tf.app.flags.DEFINE_integer('num_gpus',1,
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('batch_size', 2,
                            """Training batch size.""")
tf.app.flags.DEFINE_integer('epoch', 40,
                            """Training epoch number.""")
tf.app.flags.DEFINE_float('val_ratio', 0,
                          """Ratio of validation set when splitting dataset.""")
tf.app.flags.DEFINE_float('base_lr', 1e-3,
                          """Base learning rate.""")
tf.app.flags.DEFINE_integer('display', 1,
                            """Interval of loginfo display.""")
tf.app.flags.DEFINE_integer('stepvalue', 10000,
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
                if self.both:
                    cams[0][1, 3, 0] = cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]
                    cams[0][1, 3, 1] = -cams[0][1, 3, 1]
                    # duration = time.time() - start_time
                    # print('Back pass: d_min = %f, d_max = %f.' % \
                    #    (cams[0][1, 3, 0], cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]))
                    yield (images, cams, depth_images,ref_image)

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []

    def func(grads,g):
        grads.append(tf.expand_dims(g,0))
        return grads
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is None:
                continue
            # if tf.is_nan(g):
            #     continue
            # Add 0 dimension to the gradients to represent the tower.
            # expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            # grads=tf.cond(tf.reduce_any(tf.is_nan(g)),  lambda:grads, lambda:func(grads,g))
            grads.append(tf.expand_dims(g, 0))
        if len(grads)==0:
            continue
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def feed(iterator):
    height=FLAGS.max_h/4
    width=FLAGS.max_w/4
    # generate data
    images, cams, depth_images,ref_image= iterator.get_next()

    images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
    # gray_image.set_shape(tf.TensorShape([None,None,None,1]))
    cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
    depth_images.set_shape(tf.TensorShape([None,None, None, None, 1]))
    ref_image.set_shape(tf.TensorShape([None, None, None, 3]))
    # is_master_gpu = False
    # if i == 0:
    #     is_master_gpu = True
    depth_image = tf.squeeze(
        tf.slice(depth_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 1]), axis=1)
    depth_start = tf.reshape(
        tf.slice(cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size,1,1,1])
    depth_interval = tf.reshape(
        tf.slice(cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size,1,1,1])

    if FLAGS.regularization == 'GRU':
        prob_volume=inference_prob_recurrent(images,cams,FLAGS.max_d,depth_start,depth_interval)
        prob_map = tf.reduce_max(prob_volume,1)#b,h,w,1
        loss,_,less_one_accuracy,less_three_accuracy,depth_map=mvsnet_classification_loss(prob_volume,depth_image,FLAGS.max_d,depth_start,depth_interval)

        # regularizer_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # loss+=regularizer_loss
        # loss,less_one_accuracy,less_three_accuracy=mvsnet_regression_loss(depth_map,depth_image,depth_interval)
        # valid_mask=tf.cast(depth_image>0,tf.float32)
        # op_mask=tf.cast((tf.abs(depth_map-depth_image)<5*depth_interval)&(depth_image>0),tf.float32)
        # gt_index_volume = tf.one_hot(tf.squeeze(tf.cast(op_mask,tf.int32),-1), 2, axis=-1)#b,h,w,2
        # valid_count=tf.count_nonzero(valid_mask,dtype=tf.float32)
        # pas_mask=(1-op_mask)*valid_mask#b,h,w,1

        # cross_entropy_image = -tf.reduce_sum(gt_index_volume * tf.log(mask_volume), axis=-1,keepdims=True)*valid_mask #b,h
        # percentage=tf.count_nonzero(op_mask,dtype=tf.float32)/valid_count
        # mask_loss=tf.reduce_sum(cross_entropy_image*op_mask*percentage+cross_entropy_image*pas_mask*(1-percentage))/valid_count
        # pred_mask=tf.cast(tf.argmax(mask_volume,-1),tf.float32)
        # pred_mask=tf.expand_dims(pred_mask,-1)
        mask=tf.cast(depth_image>0,tf.float32)
        depth_map=depth_map*mask
        prob_map = prob_map*mask
    return loss,less_one_accuracy,less_three_accuracy,depth_image,depth_map,ref_image,prob_map,mask


def train(traning_list,valid_list):
    """ training mvsnet """
    training_sample_size = len(traning_list)*2
    valid_sample_size = len(valid_list)
    #if FLAGS.regularization == 'GRU':
    #if FLAGS.regularization == 'GRU':
    # training_sample_size = training_sample_size
    print('sample number: ', training_sample_size)
    print('valid sample number: ', valid_sample_size)

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        ########## data iterator #########
        # training generators
        training_generator = iter(MVSGenerator(traning_list, FLAGS.view_num,True))

        generator_data_type = (tf.float32, tf.float32,tf.float32,tf.float32)
        # dataset from generator
        training_set = tf.data.Dataset.from_generator(lambda: training_generator, generator_data_type)
        training_set = training_set.batch(FLAGS.batch_size)
        training_set = training_set.prefetch(buffer_size=FLAGS.batch_size)
        # iterators
        training_iterator = training_set.make_initializable_iterator()

        ########## optimization options ##########
        global_step = tf.Variable(0, trainable=False, name='global_step')
        valid_step=tf.Variable(0, trainable=False, name='valid_step')
        lr_op = tf.train.exponential_decay(FLAGS.base_lr, global_step=global_step,
                                           decay_steps=FLAGS.stepvalue/FLAGS.num_gpus/FLAGS.batch_size, decay_rate=FLAGS.gamma, name='lr')
        opt = tf.train.AdamOptimizer(learning_rate=lr_op )
        tower_grads = []
        less_one_accuracies=[]
        less_three_accuracies=[]

        with tf.name_scope('Model_tower' ) as scope:
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    loss,less_one_accuracy,less_three_accuracy,depth_image,depth_map,ref_image,prob_map,mask_map=feed(training_iterator)
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    # calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)
                    # keep track of the gradients across all towers.
                    tower_grads.append(grads)
                    less_one_accuracies.append(less_one_accuracy)
                    less_three_accuracies.append(less_three_accuracy)
                    less_one_accuracy=tf.reduce_mean(tf.stack(less_one_accuracies,0))
                    less_three_accuracy=tf.reduce_mean(tf.stack(less_three_accuracies,0))
                # average gradient

                grads = average_gradients(tower_grads)
                grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
                # training opt
                # train_opt=tf.cond(loss>0,lambda:opt.apply_gradients(grads, global_step=global_step),lambda:loss)
                train_opt = opt.apply_gradients(grads, global_step=global_step)
                # summary
                summaries.append(tf.summary.image("mask_map", mask_map,family="train"))
                summaries.append(tf.summary.image("prob_map", prob_map,family="train"))
                summaries.append(tf.summary.image("depth_gt", depth_image,family="train"))
                summaries.append(tf.summary.image("depth_pred", depth_map,family="train"))
                summaries.append(tf.summary.image("ref_image", ref_image,family="train"))
                summaries.append(tf.summary.scalar('loss', loss,family="train"))
                # summaries.append(tf.summary.scalar('mask_loss', mask_loss,family="train"))
                summaries.append(tf.summary.scalar('less_one_accuracy', less_one_accuracy,family="train"))
                summaries.append(tf.summary.scalar('less_three_accuracy', less_three_accuracy,family="train"))
                summaries.append(tf.summary.scalar('lr', lr_op,family="train"))
                weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                for var in weights_list:
                    summaries.append(tf.summary.histogram(var.op.name, var))
                for grad, var in grads:
                    if grad is not None:
                        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
                summary_op = tf.summary.merge(summaries)
            ## valid block
            # dataset from generator
            # valid_generator = iter(MVSGenerator(valid_list, FLAGS.view_num,False))
            # valid_set = tf.data.Dataset.from_generator(lambda: training_generator, generator_data_type)
            # valid_set = training_set.batch(FLAGS.batch_size)
            # valid_set = training_set.prefetch(buffer_size=FLAGS.batch_size)
            # # iterators
            # valid_iterator = valid_set.make_initializable_iterator()
            # with tf.device('/gpu:%d' % 0):
            #     valid_loss,valid_less_one_accuracy,valid_less_three_accuracy,valid_depth_image,valid_depth_map,valid_ref_image=feed(valid_iterator)


            # valid_summaries=[]
            # # valid_summaries.append(tf.summary.image("mask_gt", valid_mask_gt,family="valid"))
            # # valid_summaries.append(tf.summary.image("mask_pred", valid_mask_pred,family="valid"))
            # valid_summaries.append(tf.summary.image("depth_gt", valid_depth_image,family="valid"))
            # valid_summaries.append(tf.summary.image("depth_pred", valid_depth_map,family="valid"))
            # valid_summaries.append(tf.summary.image("ref_image", valid_ref_image,family="valid"))
            # valid_summaries.append(tf.summary.scalar('loss', valid_loss,family="valid"))
            # # valid_summaries.append(tf.summary.scalar('mask_loss', valid_mask_loss,family="valid"))
            # valid_summaries.append(tf.summary.scalar('less_one_accuracy', valid_less_one_accuracy,family="valid"))
            # valid_summaries.append(tf.summary.scalar('less_three_accuracy', valid_less_three_accuracy,family="valid"))
            # valid_step=tf.assign_add(valid_step,1)
            # valid_summary_op = tf.summary.merge(valid_summaries)
        # saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        # initialization option
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)

        config.gpu_options.allow_growth = True

        # config.gpu_options.set_per_process_memory_growth()
        with tf.Session(config=config) as sess:

            # initialization
            total_step = 0
            sess.run(init_op)
            summary_writer = tf.summary.FileWriter(FLAGS.logdirs, sess.graph)

            # load pre-trained model
            if FLAGS.use_pretrain:
                pretrained_model_path = os.path.join(FLAGS.model_dir,"snetgn_singlhomo_ulstm_128_both_interval_scale_1.08_bilateral_weight", "GRU", 'model.ckpt')
#                 pretrained_model_path = os.path.join("/xdata/wuyk/yjf_tf_model/19-07-26-8/GRU/", 'model.ckpt')
                # pretrained_model_path = os.path.join("/home/haibao637/", 'model.ckpt')
#
                restorer = tf.train.Saver(tf.global_variables())
                pretrained_model_path='-'.join([pretrained_model_path, str(FLAGS.ckpt_step)])
                # restorer.restore(sess, pretrained_model_path)
                optimistic_restore(sess,pretrained_model_path)

                # pretrained_model_path= '-'.join(["/home/haibao637/data5/model/tf_model/3DCNNs/model.ckpt", str(FLAGS.ckpt_step)])
                # restorer.restore (sess,pretrained_model_path )
                print(Notify.INFO, 'Pre-trained model restored from %s'%pretrained_model_path, Notify.ENDC)
                # global_step=tf.assign(global_step,0)
                # global_step=global_step.assign(0)
                # total_step =total_step*0
            # output_global_step=sess.run(global_step,feed_dict={global_step:0})
            # print("start global step:",output_global_step)
            # training several epochs
            for epoch in range(FLAGS.epoch):

                # training of one epoch
                step = 0

                sess.run(training_iterator.initializer)
                for _ in range(int(training_sample_size /(FLAGS.batch_size* FLAGS.num_gpus))):

                    # run one batch
                    start_time = time.time()
                    try:
                        out_summary_op, out_opt, out_loss, out_less_one, out_less_three = sess.run(
                            [summary_op, train_opt, loss, less_one_accuracy, less_three_accuracy])
                    except tf.errors.OutOfRangeError:
                        print("End of dataset")  # ==> "End of dataset"
                        model_folder = os.path.join(FLAGS.model_dir,model_version, FLAGS.regularization)
                        if not os.path.exists(model_folder):
                            os.makedirs(model_folder)
                        ckpt_path = os.path.join(model_folder, 'model.ckpt')
                        print(Notify.INFO, 'Saving model to %s' % ckpt_path, Notify.ENDC)
                        saver.save(sess, ckpt_path, global_step=total_step)
                        break
                    duration = time.time() - start_time
                    # print(out_depth.min(),out_depth.max())
                    # print info
                    if step % FLAGS.display == 0:
                        print(Notify.INFO,
                              'epoch, %d, step %d (%.4f), total_step %d, loss = %.4f, (< 1px) = %.4f, (< 3px) = %.4f (%.3f sec/step)' %
                              (epoch, step,float(step)/float(training_sample_size), total_step, out_loss, out_less_one, out_less_three, duration), Notify.ENDC)
                        # print(Notify.INFO,'depth min :%.4f depth max:%.4f'%(out_depth_map2.min(),out_depth_map2.max()))
                    # write summary
                    if (total_step % (FLAGS.display * 10) == 0):
                        summary_writer.add_summary(out_summary_op, total_step)

                    # save the model checkpoint periodically
                    if (total_step % FLAGS.snapshot == 0 or step == (training_sample_size - 1)):
                        model_folder = os.path.join(FLAGS.model_dir,model_version, FLAGS.regularization)
                        if not os.path.exists(model_folder):
                            os.makedirs(model_folder)
                        ckpt_path = os.path.join(model_folder, 'model.ckpt')
                        print(Notify.INFO, 'Saving model to %s' % ckpt_path, Notify.ENDC)
                        saver.save(sess, ckpt_path, global_step=total_step)
                    step += FLAGS.batch_size * FLAGS.num_gpus
                    total_step += FLAGS.batch_size * FLAGS.num_gpus

                # sess.run(valid_iterator.initializer)
                # step=0
                # for _ in range(int(valid_sample_size)):

                #     # run one batch
                #     start_time = time.time()
                #     try:
                #         out_step,out_summary_op, out_loss, out_less_one, out_less_three = sess.run(
                #             [valid_step,valid_summary_op, valid_loss, valid_less_one_accuracy, valid_less_three_accuracy])
                #     except tf.errors.OutOfRangeError:
                #         print("End of dataset")  # ==> "End of dataset"
                #         break
                #     duration = time.time() - start_time

                #     if step % FLAGS.display == 0:
                #         print(Notify.INFO,
                #               'validation at %d, step %d (%.4f), loss = %.4f, (< 1px) = %.4f, (< 3px) = %.4f (%.3f sec/step)' %
                #               (total_step, out_step,float(step)/float(valid_sample_size), out_loss, out_less_one, out_less_three, duration), Notify.ENDC)
                #     # write summary
                #     if (out_step % (FLAGS.display * 10) == 0):
                #         summary_writer.add_summary(out_summary_op, out_step)
                #     step += 1
def main(argv=None):  # pylint: disable=unused-argument
    """ program entrance """
    # Prepare all training samples
    train_sample = gen_dtu_resized_path(FLAGS.dtu_data_root)
    # # sample_list=gen_demon_list(FLAGS.demon_data_root)
    valid_sample=gen_dtu_resized_path(FLAGS.dtu_data_root,'validation')
    # sample_list1=gen_eth3d_path(FLAGS.eth3d_data_root)
    # sample_list.extend(sample_list1)
    # Shuffle
    # random.shuffle(sample_list)
    # # Training entrance.
    # # print([(sample_list[0][i].image) for i in range(3)])
    # train(sample_list,valid_list)
    # with open("train_pair.txt",'r') as f:
    #     train_sample=json.load(f)
    #     print(len(train_sample))
    # with open("valid_pair.txt",'r') as f:
    #     valid_sample=json.load(f)
    #     print(len(valid_sample))
    train(train_sample,valid_sample)
if __name__ == '__main__':
    print ('Training MVSNet with %d views' % FLAGS.view_num)
    tf.app.run()
