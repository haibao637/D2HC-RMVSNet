#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019, Yao Yao, HKUST.
Loss formulations.
"""
import queue
import sys
import math
import tensorflow as tf
import numpy as np
from tools.common import Notify
# from crfasrnn.crfrnn_layer import CrfRnnLayer
from loss import mvsnet_regression_loss

sys.path.append("../")
from cnn_wrapper.mvsnet import *
from convgru import ConvGRUCell, BGRUCell
from homography_warping import *
from lstm import *
FLAGS = tf.app.flags.FLAGS
def deconv_gn(input_tensor,
                  kernel_size,
                  filters,
                  strides,
                  name,
                  relu=False,
                  center=False,
                  scale=False,
                  channel_wise=True,
                  group=32,
                  group_channel=8,
                  padding='same',
                  biased=False,
                  reuse=tf.AUTO_REUSE):
        assert len(input_tensor.get_shape()) == 4

        # deconvolution
        res=tf.layers.conv2d_transpose(input_tensor, kernel_size=kernel_size, filters=filters, padding=padding, strides=strides,
    reuse=reuse, name=name )
        # group normalization
        x = tf.transpose(res, [0, 3, 1, 2])
        shape = tf.shape(x)
        N = shape[0]
        C = x.get_shape()[1]
        H = shape[2]
        W = shape[3]
        if channel_wise:
            G = max(1, C / group_channel)
        else:
            G = min(group, C)

        # normalization
        x = tf.reshape(x, [N, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + 1e-5)

        # per channel scale and bias (gamma and beta)
        with tf.variable_scope(name + '/gn', reuse=reuse):
            if scale:
                gamma = tf.get_variable('gamma', [C], dtype=tf.float32, initializer=tf.ones_initializer())
            else:
                gamma = tf.constant(1.0, shape=[C])
            if center:
                beta = tf.get_variable('beta', [C], dtype=tf.float32, initializer=tf.zeros_initializer())
            else:
                beta = tf.constant(0.0, shape=[C])
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])
        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta

        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])

        if relu:
            output = tf.nn.relu(output, name + '/relu')
        return output

def conv_gn(input_tensor,
            kernel_size,
            filters,
            strides,
            name,
            relu=False,
            center=False,
            scale=False,
            channel_wise=True,
            group=32,
            group_channel=8,
            padding='same',
            biased=False,
            reuse=tf.AUTO_REUSE,
            dilation=1):
        assert len(input_tensor.get_shape()) == 4

        # deconvolution
        res=tf.layers.conv2d(input_tensor, kernel_size=kernel_size, filters=filters, padding=padding, strides=strides,
    reuse=reuse, name=name ,dilation_rate=dilation)
        # group normalization
        x = tf.transpose(res, [0, 3, 1, 2])
        shape = tf.shape(x)
        N = shape[0]
        C = x.get_shape()[1]
        H = shape[2]
        W = shape[3]
        if channel_wise:
            G = max(1, C / group_channel)
        else:
            G = min(group, C)

        # normalization
        x = tf.reshape(x, [N, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + 1e-5)

        # per channel scale and bias (gamma and beta)
        with tf.variable_scope(name + '/gn', reuse=reuse):
            if scale:
                gamma = tf.get_variable('gamma', [C], dtype=tf.float32, initializer=tf.ones_initializer())
            else:
                gamma = tf.constant(1.0, shape=[C])
            if center:
                beta = tf.get_variable('beta', [C], dtype=tf.float32, initializer=tf.zeros_initializer())
            else:
                beta = tf.constant(0.0, shape=[C])
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])
        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta

        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])

        if relu:
            output = tf.nn.relu(output, name + '/relu')
        return output
def resnet_block_gn(input,kernel_size=3, filters=32, padding='same', strides=1,
    reuse=tf.AUTO_REUSE, name=None,group=32,group_channel=8 ):
    o1=conv_gn(input,kernel_size,filters,strides,relu=True,dilation=1,name=name+"_conv_0")
    o2=conv_gn(o1,kernel_size,filters,strides,relu=False,dilation=1,name=name+"_conv_1")
    return tf.nn.relu(o1+o2)

def gateNet(input,input_channel,name):
    o = conv_gn(input,kernel_size=3,filters=8,strides=1,name=name+'_gate_conv_0',reuse=tf.AUTO_REUSE)
    o = resnet_block_gn(o,kernel_size=1,filters=8,name=name)
    o = tf.layers.conv2d(o,kernel_size=3, filters=1, padding='same', strides=1,
       reuse=tf.AUTO_REUSE, name=name+"_gate_conv_1" ,dilation_rate=1)
    return tf.nn.sigmoid(o)


def inference_prob_recurrent(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer disparity image from stereo images and cameras """
    batch_size=FLAGS.batch_size
    height=FLAGS.max_h
    width=FLAGS.max_w
    ref_image=tf.squeeze(tf.slice(images,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
    images=tf.reshape(images,[-1,height,width,3])

    feature_tower=SNetDS2GN_1({'data':images}, is_training=True, reuse=tf.AUTO_REUSE)
    features=tf.reshape(feature_tower.get_output(),[FLAGS.batch_size,FLAGS.view_num,height,width,32])
    # features=tf.math.l2_normalize(features,-1)
    # features=tf.stop_gradient(features)
    ref_feature=tf.squeeze(tf.slice(features,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
    view_features=tf.slice(features,[0,1,0,0,0],[-1,-1,-1,-1,-1])

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    # reference image
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        # view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        # homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
        #                                 depth_start=depth_start, depth_interval=depth_interval)
        # view_homographies.append(homographies)
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        if FLAGS.inverse_depth:
            homographies = get_homographies_inv_depth(ref_cam, view_cam, depth_num=depth_num,
                                depth_start=depth_start, depth_end=depth_end)
        else:
            homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                            depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    gru1_filters = 16
    gru2_filters = 4
    gru3_filters = 2
    feature_shape = [FLAGS.batch_size, FLAGS.max_h, FLAGS.max_w, 32]
    batch_size,height,width,channel=feature_shape
    cell0=ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height, width,32],
        output_channels=16,
        kernel_shape=[3, 3],

        name="conv_lstm_cell0"
    )
    cell1=ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height/2, width/2, 16],
        output_channels=16,
        kernel_shape=[3, 3],
        name="conv_lstm_cell1"
    )

    cell2=ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height/4, width/4, 16],
        output_channels=16,
        kernel_shape=[3, 3],
        name="conv_lstm_cell2"
    )

    cell3=ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height/2, width/2, 32],
        output_channels=16,
        kernel_shape=[3, 3],
        name="conv_lstm_cell3"
    )

    cell4=ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height, width, 32],
        output_channels=8,
        kernel_shape=[3, 3],
        name="conv_lstm_cell4"
    )

    initial_state0 = cell0.zero_state(batch_size, dtype=tf.float32)
    initial_state1 = cell1.zero_state(batch_size, dtype=tf.float32)
    initial_state2 = cell2.zero_state(batch_size, dtype=tf.float32)
    initial_state3 = cell3.zero_state(batch_size, dtype=tf.float32)
    initial_state4 = cell4.zero_state(batch_size, dtype=tf.float32)


    with tf.name_scope('cost_volume_homography') as scope:

        # forward cost volume
        # depth=depth_start
        costs=[]
        # ref_feature=ref_tower.get_output()
        # ref_feature=tf.reshape(ref_feature,[batch_size,height,width,channel])
        depth_maps=[]
        # weights=tf.reshape(tf.constant([1.0,0.2,0.1],dtype=tf.float32),[1,1,1,1,3])
        # ref_feature=tf.extract_image_patches(images=ref_feature, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1],
        #                                         rates=[1, 7, 7, 1], padding='SAME')#b,h,w,9*c

        for d in range(depth_num):

            # compute cost (variation metric)
            ave_feature =ref_feature+0.0

            ave_feature2 = tf.square(ave_feature)
            warped_view_volumes = tf.zeros([batch_size,height,width,1])
            weight_sum          = tf.zeros([batch_size,height,width,1])
            for view in range(0, FLAGS.view_num - 1):

                view_feature=tf.squeeze(tf.slice(view_features,[0,view,0,0,0],[-1,1,-1,-1,-1]),1)
                homographies = view_homographies[view]
                homographies = tf.transpose(homographies, perm=[1, 0, 2, 3])
                homography = homographies[d]
                warped_view_feature = tf_transform_homography(view_feature, homography)
                warped_view_volume = tf.square(warped_view_feature-ref_feature)
                weight = gateNet(warped_view_volume,32,name='gate')
                warped_view_volumes += (weight+1)*warped_view_volume
                weight_sum          += (weight+1)

            cost=warped_view_volumes/weight_sum #= ave_feature2 - tf.square(ave_feature)


            with tf.variable_scope("rnn/", reuse=tf.AUTO_REUSE):
                cost0,initial_state0=cell0(cost,state=initial_state0)
                cost1=tf.nn.max_pool2d(cost0,(2,2),2,'SAME')
                cost1,initial_state1=cell1(cost1,state=initial_state1)
                cost2=tf.nn.max_pool2d(cost1,(2,2),2,'SAME')
                cost2,initial_state2=cell2(cost2,state=initial_state2)
                cost2=deconv_gn(cost2, 16, 3, padding='same',strides=2, reuse=tf.AUTO_REUSE, name='cost_upconv0' )

                cost2=tf.concat([cost2,cost1],-1)
                cost3,initial_state3=cell3(cost2,state=initial_state3)
                cost3=deconv_gn(cost3, 16, 3, padding='same',strides=2, reuse=tf.AUTO_REUSE, name='cost_upconv1' )
                cost3=tf.concat([cost3,cost0],-1)
                cost4,initial_state4=cell4(cost3,state=initial_state4)
            cost = tf.layers.conv2d(
                cost4, 1, 3, padding='same', reuse=tf.AUTO_REUSE, name='prob_conv')

            costs.append(cost)

        prob_volume = tf.stack(costs, axis=1)

        prob_volume = tf.nn.softmax(-prob_volume, axis=1, name='prob_volume')
        # depth_maps=tf.stack(depth_maps,1)

        # depth_map=tf.reduce_sum((prob_volume*depth_maps),1)#b,h,w,1
    return prob_volume


def inference_winner_take_all(images, cams, depth_num, depth_start, depth_end,
                              is_master_gpu=True, reg_type='GRU', inverse_depth=False):
    """ infer disparity image from stereo images and cameras """

    if not inverse_depth:
        depth_interval = (depth_end - depth_start) / (tf.cast(depth_num, tf.float32) - 1)

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)


    height=tf.shape(images)[2]
    width=tf.shape(images)[3]
    images=tf.reshape(images,[-1,height,width,3])
    feature_tower=SNetDS2GN_1({'data':images}, is_training=True, reuse=tf.AUTO_REUSE,dilation=1).get_output()
    height=tf.shape(feature_tower)[1]
    width=tf.shape(feature_tower)[2]

    features=tf.reshape(feature_tower,[FLAGS.batch_size,FLAGS.view_num,height,width,32])
    # features=tf.math.l2_normalize(features,-1)
    # features=tf.stop_gradient(features)
    ref_feature=tf.squeeze(tf.slice(features,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
    view_features=tf.slice(features,[0,1,0,0,0],[-1,-1,-1,-1,-1])


    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        if inverse_depth:
            homographies = get_homographies_inv_depth(ref_cam, view_cam, depth_num=depth_num,
                                depth_start=depth_start, depth_end=depth_end)
        else:
            homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                            depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    # gru unit
    gru1_filters = 16
    gru2_filters = 4
    gru3_filters = 2
    feature_shape = [FLAGS.batch_size, FLAGS.max_h, FLAGS.max_w, 32]
    batch_size,height,width,channel=feature_shape
    gru_input_shape = [feature_shape[1], feature_shape[2]]

    cell0=ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height, width,32],
        output_channels=16,
        kernel_shape=[3, 3],

        name="conv_lstm_cell0"
    )
    cell1=ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height/2, width/2, 16],
        output_channels=16,
        kernel_shape=[3, 3],
        name="conv_lstm_cell1"
    )

    cell2=ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height/4, width/4, 16],
        output_channels=16,
        kernel_shape=[3, 3],
        name="conv_lstm_cell2"
    )

    cell3=ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height/2, width/2, 32],
        output_channels=16,
        kernel_shape=[3, 3],
        name="conv_lstm_cell3"
    )

    cell4=ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height, width, 32],
        output_channels=8,
        kernel_shape=[3, 3],
        name="conv_lstm_cell4"
    )


    initial_state0 = cell0.zero_state(batch_size, dtype=tf.float32)
    initial_state1 = cell1.zero_state(batch_size, dtype=tf.float32)
    initial_state2 = cell2.zero_state(batch_size, dtype=tf.float32)
    initial_state3 = cell3.zero_state(batch_size, dtype=tf.float32)
    initial_state4 = cell4.zero_state(batch_size, dtype=tf.float32)

    # initialize variables
    exp_sum = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
        name='exp_sum', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    depth_image = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
        name='depth_image', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    max_prob_image = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
        name='max_prob_image', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    init_map = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], 1])
    weights=tf.reshape(tf.constant([1.0,0.5,0.1],dtype=tf.float32),[1,1,1,1,3])
    # define winner take all loop
    def body(depth_index, initial_state0, initial_state1, initial_state2, initial_state3, initial_state4,  depth_image, max_prob_image, exp_sum, incre):
        """Loop body."""

        # calculate cost
        ave_feature =ref_feature
        ave_feature2 = tf.square(ref_feature)

        warped_view_volumes = tf.zeros([batch_size,height,width,1])
        weight_sum          = tf.zeros([batch_size,height,width,1])
        for view in range(0, FLAGS.view_num - 1):

            homographies = view_homographies[view]
            homographies = tf.transpose(homographies, perm=[1, 0, 2, 3])
            homography = homographies[depth_index]
            view_feature=tf.squeeze(tf.slice(view_features,[0,view,0,0,0],[-1,1,-1,-1,-1]),1)
            warped_view_feature = tf_transform_homography(view_feature, homography)
            warped_view_volume = tf.square(warped_view_feature-ref_feature)
            weight = gateNet(warped_view_volume,32,name='gate')
            warped_view_volumes += (weight+1)*warped_view_volume
            weight_sum          += (weight+1)

        cost=warped_view_volumes/weight_sum
        # cost=tf.expand_dims(cost,1)

        with  tf.name_scope('cost_volume_homography') as scope:
            # with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            print(Notify.INFO,tf.shape(cost),Notify.ENDC)

            with tf.variable_scope("rnn/", reuse=tf.AUTO_REUSE):
                cost0,initial_state0=cell0(cost,state=initial_state0)
                cost1=tf.nn.max_pool2d(cost0,(2,2),2,'SAME')
                cost1,initial_state1=cell1(cost1,state=initial_state1)
                cost2=tf.nn.max_pool2d(cost1,(2,2),2,'SAME')
                cost2,initial_state2=cell2(cost2,state=initial_state2)
                cost2=deconv_gn(cost2, 16, 3, padding='same',strides=2, reuse=tf.AUTO_REUSE, name='cost_upconv0' )

                cost2=tf.concat([cost2,cost1],-1)
                cost3,initial_state3=cell3(cost2,state=initial_state3)
                cost3=deconv_gn(cost3, 16, 3, padding='same',strides=2, reuse=tf.AUTO_REUSE, name='cost_upconv1' )
                cost3=tf.concat([cost3,cost0],-1)
                cost4,initial_state4=cell4(cost3,state=initial_state4)

            cost = tf.layers.conv2d(
                cost4, 1, 3, padding='same', reuse=tf.AUTO_REUSE, name='prob_conv')
            prob = tf.exp(-cost)

        # index
        d_idx = tf.cast(depth_index, tf.float32)
        if inverse_depth:
            inv_depth_start = tf.div(1.0, depth_start)
            inv_depth_end = tf.div(1.0, depth_end)
            inv_interval = (inv_depth_start - inv_depth_end) / (tf.cast(depth_num, 'float32') - 1)
            inv_depth = inv_depth_start - d_idx * inv_interval
            depth = tf.div(1.0, inv_depth)
        else:
            depth = depth_start + d_idx * depth_interval
        temp_depth_image = tf.reshape(depth, [FLAGS.batch_size, 1, 1, 1])
        temp_depth_image = tf.tile(
            temp_depth_image, [1, feature_shape[1], feature_shape[2], 1])

        # update the best
        update_flag_image = tf.cast(tf.less(max_prob_image, prob), dtype='float32')
        new_max_prob_image = update_flag_image * prob + (1 - update_flag_image) * max_prob_image
        new_depth_image = update_flag_image * temp_depth_image + (1 - update_flag_image) * depth_image
        max_prob_image = tf.assign(max_prob_image, new_max_prob_image)
        depth_image = tf.assign(depth_image, new_depth_image)

        # update counter
        exp_sum = tf.assign_add(exp_sum, prob)
        depth_index = tf.add(depth_index, incre)

        return depth_index, initial_state0, initial_state1, initial_state2,  initial_state3, initial_state4, depth_image, max_prob_image, exp_sum, incre

    # run forward loop
    exp_sum = tf.assign(exp_sum, init_map)
    depth_image = tf.assign(depth_image, init_map)
    max_prob_image = tf.assign(max_prob_image, init_map)
    depth_index = tf.constant(0)
    incre = tf.constant(1)
    cond = lambda depth_index, *_: tf.less(depth_index, depth_num)
    _, initial_state0, initial_state1, initial_state2, initial_state3, initial_state4,  depth_image, max_prob_image, exp_sum, incre = tf.while_loop(
        cond, body
        , [depth_index, initial_state0, initial_state1, initial_state2,initial_state3, initial_state4,  depth_image, max_prob_image, exp_sum, incre]
        , back_prop=False, parallel_iterations=1)

    # get output
    forward_exp_sum = exp_sum + 1e-7
    forward_depth_map = depth_image
    return forward_depth_map, max_prob_image / forward_exp_sum
