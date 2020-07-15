#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Model architectures.
"""

import sys
import math
import tensorflow as tf
import numpy as np
from homography_warping import *
FLAGS = tf.app.flags.FLAGS

def non_zero_mean_absolute_diff(y_true, y_pred, interval):
    """ non zero mean absolute loss for one batch """
    with tf.name_scope('MAE'):
        shape = tf.shape(y_pred)
        # interval = tf.reshape(interval, [])
        interval = tf.reshape(interval, [-1, 1, 1, 1])
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true) + 1e-7
        masked_abs_error = tf.abs(mask_true * (y_true - y_pred)/interval)            # 4D
        masked_mae = tf.reduce_sum(masked_abs_error, axis=[1, 2, 3])        # 1D
        masked_mae = tf.reduce_sum((masked_mae) / denom)         # 1
        # masked_mae=tf.reduce_mean(tf.square(y_true-y_pred))
    return masked_mae

def less_one_percentage(y_true, y_pred, interval):
    """ less one accuracy for one batch """
    with tf.name_scope('less_one_error'):
        shape = tf.shape(y_pred)
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true) + 1e-7
        interval_image = tf.reshape(interval, [-1, 1, 1, 1])
        abs_diff_image = tf.abs(y_true - y_pred) / interval_image
        less_one_image = mask_true * tf.cast(tf.less_equal(abs_diff_image, 1.0), dtype='float32')
    return tf.reduce_sum(less_one_image) / denom

def less_three_percentage(y_true, y_pred, interval):
    """ less three accuracy for one batch """
    with tf.name_scope('less_three_error'):
        shape = tf.shape(y_pred)
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true) + 1e-7
        interval_image = tf.reshape(interval, [-1, 1, 1, 1])
        abs_diff_image = tf.abs(y_true - y_pred) / interval_image
        less_three_image = mask_true * tf.cast(tf.less_equal(abs_diff_image, 3.0), dtype='float32')
    return tf.reduce_sum(less_three_image) / denom

def mvsnet_regression_loss(estimated_depth_image, depth_image, depth_interval):
    """ compute loss and accuracy """
    # non zero mean absulote loss
    masked_mae = non_zero_mean_absolute_diff(depth_image, estimated_depth_image, tf.abs(depth_interval))
    # less one accuracy
    less_one_accuracy = less_one_percentage(depth_image, estimated_depth_image, tf.abs(depth_interval))
    # less three accuracy
    less_three_accuracy = less_three_percentage(depth_image, estimated_depth_image, tf.abs(depth_interval))

    return masked_mae, less_one_accuracy, less_three_accuracy
def normal_loss(depth_map,depth_image,K):
    """
    depth_map: b,h,w,1
    depth_image:b,h,w,1
    K: camera intrinsic matrix b,3,3
    """
    shape=tf.shape(depth_map)
    batch_size=shape[0]
    height=shape[1]
    width=shape[2]
    pixel_grids = get_pixel_grids(height, width)
    pixel_grids = tf.expand_dims(pixel_grids, 0)
    pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
    pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))#b,3,-1
    depth_raw=tf.reshape(depth_map,[batch_size,1,-1])
    pixel_map_raw=pixel_grids*depth_raw
    inv_K=tf.matrix_inverse(K)
    pixel_map_raw = tf.matmul(inv_K, pixel_map_raw)#b,3,-1
    pixel_map_raw=tf.transpose(tf.reshape(pixel_map_raw,[batch_size,3,height,width]),[0,2,3,1])

    gt_raw=tf.reshape(depth_image,[batch_size,1,-1])
    pixel_gt_raw=pixel_grids*gt_raw
    pixel_gt_raw=tf.matmul(K,pixel_gt_raw)#b,3,-1
    pixel_gt_raw=tf.transpose(tf.reshape(pixel_gt_raw,[batch_size,3,height,width]),[0,2,3,1])

    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.tile(tf.reshape(sobel_x, [3, 3, 1, 1]),[1,1,3,3])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])
    
    # Shape = 1x height x width x 3.
    pixel_gt_raw_x = tf.nn.conv2d(pixel_gt_raw, sobel_x_filter,
                            strides=[1, 1, 1, 1], padding='SAME')
    pixel_gt_raw_y = tf.nn.conv2d(pixel_gt_raw, sobel_y_filter,
                            strides=[1, 1, 1, 1], padding='SAME')

    pixel_map_raw_x = tf.nn.conv2d(pixel_map_raw, sobel_x_filter,
                            strides=[1, 1, 1, 1], padding='SAME')
    pixel_map_raw_y = tf.nn.conv2d(pixel_map_raw, sobel_y_filter,
                            strides=[1, 1, 1, 1], padding='SAME')
    
    normal_map=tf.cross(pixel_map_raw_x,pixel_map_raw_y)#b,h,w,3
    normal_gt=tf.cross(pixel_gt_raw_x,pixel_gt_raw_y)#b,h,w,3
    normal_map=tf.nn.l2_normalize(normal_map,axis=-1,epsilon=1e-12)
    normal_gt=tf.nn.l2_normalize(normal_gt,axis=-1,epsilon=1e-12)
    loss_map=tf.reduce_sum((normal_map-normal_gt)**2,axis=-1,keepdims=True)
    mask_true=tf.cast(depth_image>0.0,tf.float32)
    valid_pixel_num = tf.reduce_sum(mask_true) + 1e-7
    loss_map=loss_map*mask_true
    loss=tf.reduce_sum(loss_map)/valid_pixel_num
    return loss


# def mvsnet_crf_loss(prob_volume,gray_image,gt_depth_image,depth_num,depth_start,depth_interval):
#     """ compute loss and accuracy """
#
#     # get depth mask
#
#     mask_true = tf.cast(tf.not_equal(gt_depth_image, 0.0), dtype='float32')
#     valid_pixel_num = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
#     # gt depth map -> gt index map
#     shape = tf.shape(gt_depth_image)
#     depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
#     start_mat = tf.tile(tf.reshape(depth_start, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
#
#     interval_mat = tf.tile(tf.reshape(depth_interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
#     # gt_index_image = tf.div(gt_depth_image - start_mat, interval_mat)
#     # gt_index_image = tf.multiply(mask_true, gt_index_image)+tf.multiply(1-mask_true,FLAGS.max_d+1)
#     # gt_index_image = tf.cast(tf.round(gt_index_image), dtype='int32')
#     # gt index map -> gt one hot volume (B x H x W x 1)
#     # gt_index_volume = tf.one_hot(gt_index_image, depth_num, axis=1)
#     prob_volume=tf.transpose(tf.squeeze(prob_volume,-1),[0,2,3,1])#b,h,w,d
#     height=FLAGS.max_h/4
#     width=FLAGS.max_w/4
#     prob_volume=tf.reshape(prob_volume,[FLAGS.batch_size,height*width,FLAGS.max_d+1])
#     lengths = tf.cast(tf.ones([FLAGS.batch_size]) * height*width, tf.int32)
#     gray_image=tf.reshape(gray_image,[FLAGS.batch_size,height*width])
#     log_likehood, transition_params = tf.contrib.crf.crf_log_likelihood(
#         inputs=prob_volume,
#         tag_indices=gray_image,
#         sequence_lengths=lengths
#     )
#
#     loss=tf.reduce_mean(-log_likehood)
#     decode_tags,best_score=tf.contrib.crf.crf_decode(prob_volume,transition_params,lengths)
#     wta_index_map=tf.reshape(tf.cast(decode_tags,tf.float32),[FLAGS.batch_size,height,width,1])
#     mask=tf.cast(wta_index_map<(FLAGS.max_d+1),tf.float32)
#     wta_depth_map = mask*(wta_index_map * interval_mat + start_mat)
#     # less one accuracy
#     less_one_accuracy = less_one_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
#     # less three accuracy
#     less_three_accuracy = less_three_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
#     return loss,less_one_accuracy,less_three_accuracy

# def mvsnet_classification_loss(prob_volume, gt_depth_image, depth_num, depth_start, depth_interval):
#     """ compute loss and accuracy """

#     # get depth mask
#     mask_true = tf.cast(tf.not_equal(gt_depth_image, 0.0), dtype='float32')
#     valid_pixel_num = tf.reduce_sum(mask_true) + 1e-7
#     # gt depth map -> gt index map
#     shape = tf.shape(gt_depth_image)
#     depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
#     start_mat = tf.reshape(depth_start, [-1, 1, 1, 1])

#     interval_mat = tf.reshape(depth_interval, [-1, 1, 1, 1])
#     gt_index_image = tf.div(gt_depth_image - start_mat, interval_mat)
#     gt_index_image = tf.multiply(mask_true, gt_index_image)
#     # depth_index_image=tf.cast(tf.argmax(prob_volume,1),tf.float32)#b,h,w,1
#     # depth_index_image=tf.multiply(mask_true,depth_index_image)
#     # expand_depth_map = tf.extract_image_patches(images=depth_index_image, ksizes=[1, 7, 7, 1], strides=[1, 1, 1, 1],
#     #                                             rates=[1, 1, 1, 1], padding='SAME')  # b,h,w,9
#     # expand_gt_map = tf.extract_image_patches(images=gt_index_image, ksizes=[1, 7, 7, 1], strides=[1, 1, 1, 1],
#     #                                          rates=[1, 1, 1, 1], padding='SAME')  # b,h,w,9
#     # Ed=expand_depth_map-tf.reduce_mean(expand_depth_map,axis=-1,keepdims=True)
#     # Eg=expand_gt_map-tf.reduce_mean(expand_gt_map,axis=-1,keepdims=True)
#     # sig=tf.sqrt(tf.reduce_mean(Ed**2,axis=-1,keepdims=True)*tf.reduce_mean(Eg**2,axis=-1,keepdims=True))+1e-7
#     gt_index_image = tf.cast(tf.round(gt_index_image), dtype='int32')
#     # # none_zero=tf.count_nonzero(mask_true)
#     # loss1=tf.reduce_sum(mask_true*tf.reduce_mean(1-Ed*Eg/sig,axis=-1,keepdims=True))/valid_pixel_num
#     # gt index map -> gt one hot volume (B x H x W x 1)
#     gt_index_volume = tf.one_hot(gt_index_image, depth_num, axis=1)
#     # cross entropy image (B x H x W x 1)
#     cross_entropy_image = -tf.reduce_sum(gt_index_volume * tf.log(prob_volume), axis=1)
#     # masked cross entropy loss
#     masked_cross_entropy_image = tf.multiply(mask_true, cross_entropy_image)
#     masked_cross_entropy = tf.reduce_sum(masked_cross_entropy_image, axis=[1, 2, 3])
#     masked_cross_entropy = tf.reduce_sum(masked_cross_entropy / valid_pixel_num)

#     # winner-take-all depth map
#     wta_index_map = tf.cast(tf.argmax(prob_volume, axis=1), dtype='float32')
#     wta_depth_map = wta_index_map * interval_mat + start_mat    

#     # non zero mean absulote loss
#     masked_mae = non_zero_mean_absolute_diff(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
#     # less one accuracy
#     less_one_accuracy = less_one_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
#     # less three accuracy
#     less_three_accuracy = less_three_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
#     return masked_cross_entropy, masked_mae, less_one_accuracy, less_three_accuracy, wta_depth_map

# def mvsnet_classification_loss(prob_volume, gt_depth_image, depth_num, depth_start, depth_interval):
#     """ compute loss and accuracy """

    # get depth mask
    # mask_true = tf.cast(tf.not_equal(gt_depth_image, 0.0), dtype='float32')
    # valid_pixel_num = tf.reduce_sum(mask_true) + 1e-7
    # # gt depth map -> gt index map
    # shape = tf.shape(gt_depth_image)
    # depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    # start_mat = tf.tile(tf.reshape(depth_start, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])

    # interval_mat = tf.tile(tf.reshape(depth_interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
    # gt_index_image = tf.div(gt_depth_image - start_mat, interval_mat)
    # gt_index_image=mask_true*gt_index_image
    # # gt_index_image = tf.multiply(mask_true, gt_index_image)
    # gt_index_image = tf.cast(tf.round(gt_index_image), dtype='int32')
    
    # # gt index map -> gt one hot volume (B x H x W x 1)
    # gt_index_volume = tf.one_hot(gt_index_image, depth_num, axis=1)
    # # cross entropy image (B x H x W x 1)
    # cross_entropy_image = -tf.reduce_sum(gt_index_volume * tf.log(prob_volume), axis=1)
    # # masked cross entropy loss
    # masked_cross_entropy_image = tf.multiply(mask_true, cross_entropy_image)
    # # masked_cross_entropy = tf.reduce_sum(masked_cross_entropy_image, axis=[1, 2, 3])
    # # masked_cross_entropy = tf.reduce_sum(masked_cross_entropy / valid_pixel_num)
    # # cross_entropy_image=cross_entropy_image*(1.0-0.99*(1-mask_true))
    # masked_cross_entropy=tf.reduce_sum(cross_entropy_image)/valid_pixel_num
    # # winner-take-all depth map
    # wta_index_map = tf.cast(tf.argmax(prob_volume, axis=1), dtype='float32')
    # wta_depth_map = wta_index_map * interval_mat + start_mat    
    # mask=tf.cast(wta_index_map<tf.cast(depth_num,tf.float32),tf.float32)
    # wta_depth_map=wta_depth_map*mask
    # # non zero mean absulote loss
    # masked_mae = non_zero_mean_absolute_diff(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
    # # less one accuracy
    # less_one_accuracy = less_one_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
    # # less three accuracy
    # less_three_accuracy = less_three_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))

    # return masked_cross_entropy, masked_mae, less_one_accuracy, less_three_accuracy, wta_depth_map

def mvsnet_classification_loss(prob_volume, gt_depth_image, depth_num, depth_start, depth_interval):
    """ compute loss and accuracy """

    # get depth mask
    mask_true = tf.cast(tf.not_equal(gt_depth_image, 0.0), dtype='float32')
    valid_pixel_num = tf.reduce_sum(mask_true) + 1e-7
    # gt depth map -> gt index map
    shape = tf.shape(gt_depth_image)
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    start_mat = tf.tile(tf.reshape(depth_start, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])

    interval_mat = tf.tile(tf.reshape(depth_interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
    gt_index_image = tf.div(tf.abs(gt_depth_image - start_mat), tf.abs(interval_mat))
    gt_index_image = tf.multiply(mask_true, gt_index_image)
    gt_index_image = tf.cast(tf.round(gt_index_image), dtype='int32')
    # gt index map -> gt one hot volume (B x H x W x 1)
    gt_index_volume = tf.one_hot(gt_index_image, depth_num, axis=1)
    # cross entropy image (B x H x W x 1)
    cross_entropy_image = -tf.reduce_sum(gt_index_volume * tf.log(prob_volume), axis=1)
    # masked cross entropy loss
    masked_cross_entropy_image = tf.multiply(mask_true, cross_entropy_image)
    masked_cross_entropy = tf.reduce_sum(masked_cross_entropy_image, axis=[1, 2, 3])
    masked_cross_entropy = tf.reduce_sum(masked_cross_entropy / valid_pixel_num)

    # winner-take-all depth map
    wta_index_map = tf.cast(tf.argmax(prob_volume, axis=1), dtype='float32')
    wta_depth_map = wta_index_map * interval_mat + start_mat    

    # non zero mean absulote loss
    masked_mae = non_zero_mean_absolute_diff(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
    # less one accuracy
    less_one_accuracy = less_one_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
    # less three accuracy
    less_three_accuracy = less_three_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))

    return masked_cross_entropy, masked_mae, less_one_accuracy, less_three_accuracy, wta_depth_map