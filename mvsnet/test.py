#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019, Yao Yao, HKUST.
Test script.
"""

from __future__ import print_function

import os
import time
import sys
import math
import argparse
import numpy as np
# from scipy.spatial.transform import Rotation as R
import cv2
import tensorflow as tf
from  tensorflow.contrib.opt import ScipyOptimizerInterface
sys.path.append("../")
from tools.common import Notify
from preprocess import *
from model import *
from loss import *
from flowmodel import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# dataset parameters
tf.app.flags.DEFINE_string('dense_folder', "/home/haibao637/data/mvsnet_input/family/",
                           """Root path to dense folder.""")
tf.app.flags.DEFINE_string('output_dir', "depths_mvsnet_1",
                           """Root path to dense folder.""")
tf.app.flags.DEFINE_string('model_dir',
                           '/home/haibao637/data/tf_models',
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step',50000,
                            """ckpt step.""")
tf.app.flags.DEFINE_float('disp_size', 50.0,
                            """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_bool('hist', True,
                            """Downsample scale for building cost volume.""")
# input parameters
tf.app.flags.DEFINE_integer('view_num',7,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 256,
                            """Maximum depth step when testing.""")
tf.app.flags.DEFINE_integer('max_w', 960,
                            """Maximum image width when testing.""")
tf.app.flags.DEFINE_integer('max_h', 512,
                            """Maximum image height when testing.""")
tf.app.flags.DEFINE_float('sample_scale', 1.0,
                            """Downsample scale for building cost volume (W and H).""")
tf.app.flags.DEFINE_float('interval_scale', 0.8,
                            """Downsample scale for building cost volume (D).""")
tf.app.flags.DEFINE_float('base_image_size', 8,
                            """Base image size""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Testing batch size.""")
tf.app.flags.DEFINE_bool('adaptive_scaling', True,
                            """Let image size to fit the network, including 'scaling', 'cropping'""")

# network architecture
tf.app.flags.DEFINE_string('regularization', 'GRU',
                           """Regularization method, including '3DCNNs' and 'GRU'""")
tf.app.flags.DEFINE_boolean('refinement', False,
                           """Whether to apply depth map refinement for MVSNet""")
tf.app.flags.DEFINE_bool('inverse_depth', True,
                           """Whether to apply inverse depth for R-MVSNet""")

FLAGS = tf.app.flags.FLAGS

class MVSGenerator:
    """ data generator class, tf only accept generator without param """
    def __init__(self, sample_list, view_num):
        self.sample_list = sample_list
        self.view_num = view_num
        self.sample_num = len(sample_list)
        self.counter = 0

    def __iter__(self):
        while True:
            for data in self.sample_list:

                # read input data
                images = []
                cams = []
                image_index = int(os.path.splitext(os.path.basename(data[0].image))[0])
                selected_view_num = int(len(data))

                for view in range(min(self.view_num, selected_view_num)):
#                     image_file = file_io.FileIO(data[2 * view], mode='r')
#                     image = scipy.misc.imread(image_file, mode='RGB')
#                     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image=cv2.imread(data[view].image)
                    cam_file = file_io.FileIO(data[view].cam, mode='r')
                    cam = load_cam(cam_file, FLAGS.interval_scale)
                    if cam[1][3][2] == 0:
                        cam[1][3][2] = FLAGS.max_d
                    images.append(image)
                    cams.append(cam)

                if selected_view_num < self.view_num:
                    for view in range(selected_view_num, self.view_num):
                        image_file = file_io.FileIO(data[0].image, mode='r')
                        image = scipy.misc.imread(image_file, mode='RGB')
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        image=cv2.imread(data[view].image)
                        cam_file = file_io.FileIO(data[view].cam, mode='r')
                        cam = load_cam(cam_file, FLAGS.interval_scale)
                        images.append(image)
                        cams.append(cam)
                print ('range: ', cams[0][1, 3, 0], cams[0][1, 3, 1], cams[0][1, 3, 2], cams[0][1, 3, 3])

                # determine a proper scale to resize input
                resize_scale = 1
                if FLAGS.adaptive_scaling:
                    h_scale = 0
                    w_scale = 0
                    for view in range(self.view_num):
                        height_scale = float(FLAGS.max_h) / images[view].shape[0]
                        width_scale = float(FLAGS.max_w) / images[view].shape[1]
                        if height_scale > h_scale:
                            h_scale = height_scale
                        if width_scale > w_scale:
                            w_scale = width_scale
                    if h_scale > 1 or w_scale > 1:
                        print ("max_h, max_w should < W and H!")
                        exit(-1)
                    resize_scale = h_scale
                    if w_scale > h_scale:
                        resize_scale = w_scale

                scaled_input_images, scaled_input_cams = scale_mvs_input(images, cams, scale=resize_scale)

                # crop to fit network
                croped_images, croped_cams = crop_mvs_input(scaled_input_images, scaled_input_cams)
                print(croped_images[0].shape)
                # center images
                centered_images = []
                for view in range(self.view_num):
                    centered_images.append(center_image(croped_images[view]))

                # sample cameras for building cost volume
                real_cams = np.copy(croped_cams)
                scaled_cams = scale_mvs_camera(croped_cams, scale=FLAGS.sample_scale)

                # return mvs input
                scaled_images = []
                for view in range(self.view_num):
                    scaled_images.append(scale_image(croped_images[view], scale=FLAGS.sample_scale))
                scaled_images = np.stack(scaled_images, axis=0)
                croped_images = np.stack(croped_images, axis=0)
                scaled_cams = np.stack(scaled_cams, axis=0)
                self.counter += 1
                # print(centered_images[0].shape)
                yield (scaled_images, centered_images, scaled_cams,real_cams, image_index)

def mvsnet_pipeline(mvs_list):

    """ mvsnet in altizure pipeline """
    print ('sample number: ', len(mvs_list))
     # GPU grows incrementally
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.InteractiveSession(config=config)
    # create output folder
    output_folder = os.path.join(FLAGS.dense_folder, FLAGS.output_dir)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # testing set
    mvs_generator = iter(MVSGenerator(mvs_list, FLAGS.view_num))
    generator_data_type = (tf.float32, tf.float32, tf.float32,tf.float32, tf.int32)
    mvs_set = tf.data.Dataset.from_generator(lambda: mvs_generator, generator_data_type)
    mvs_set = mvs_set.batch(FLAGS.batch_size)
    mvs_set = mvs_set.prefetch(buffer_size=1)

    # data from dataset via iterator
    mvs_iterator = mvs_set.make_initializable_iterator()

    scaled_images, centered_images, scaled_cams,original_cams, image_index = mvs_iterator.get_next()

    # set shapes
    scaled_images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
    centered_images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
    scaled_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
    depth_start = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_interval = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_num = tf.cast(
        tf.reshape(tf.slice(scaled_cams, [0, 0, 1, 3, 2], [1, 1, 1, 1, 1]), []), 'int32')

    # deal with inverse depth
    if FLAGS.regularization == '3DCNNs' and FLAGS.inverse_depth:
        depth_end = tf.reshape(
            tf.slice(scaled_cams, [0, 0, 1, 3, 3], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    else:
        depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # depth map inference using 3DCNNs
    if FLAGS.regularization == '3DCNNs':
        init_depth_map, prob_map = inference_mem(
            centered_images, scaled_cams, FLAGS.max_d, depth_start, depth_interval)

        if FLAGS.refinement:
            ref_image = tf.squeeze(tf.slice(centered_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
            refined_depth_map = depth_refine(
                init_depth_map, ref_image, FLAGS.max_d, depth_start, depth_interval, True)

    # depth map inference using GRU
    elif FLAGS.regularization == 'GRU':
        width=FLAGS.max_w/4
        height=FLAGS.max_h/4
        batch_size=FLAGS.batch_size
        # with tf.name_scope("Model_tower"):
        with tf.device("/gpu:0"):
            init_depth_map, prob_map = inference_winner_take_all(centered_images, scaled_cams,
                FLAGS.max_d, depth_start, depth_end, reg_type='GRU', inverse_depth=FLAGS.inverse_depth)
        # init_depth_map,prob_map=inference_mem_1(centered_images,scaled_cams,FLAGS.max_d,depth_start,depth_interval)
        # init_depth_map,prob_map=inference_mem_1(centered_images,scaled_cams,FLAGS.max_d,depth_start,depth_interval)

        # init_depth_map,_,_=depth_inference(centered_images,scaled_cams)
#         flow1,flow2,flow3,init_depth_map,probs=depth_inference_mem(centered_images,scaled_cams)
        # depths=tf.concat(depths,-1)
        # init_depth_map=tf.reduce_mean(depths,-1,keepdims=True)
        # ref_cam=tf.squeeze(tf.slice(original_cams,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
        # ref_cam=tf.stop_gradient(ref_cam)
        # warps=[]
        # # init_depth_map=depths
        # width=FLAGS.max_w
        # height=FLAGS.max_h
        # view_cam=tf.squeeze(tf.slice(original_cams,[0,1,0,0,0],[-1,1,-1,-1,-1]),1)
        # p,q=PQ(ref_cam,view_cam,[batch_size,height,width,1])

        # x_linspace = tf.linspace(0., tf.cast(width, 'float32'), width)
        # y_linspace = tf.linspace(0., tf.cast(height, 'float32'), height)
        # x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        # x_coordinates=tf.tile(tf.reshape(x_coordinates,[1,height,width,1]),[batch_size,1,1,1])
        # xcoords=x_coordinates+flow3
        # init_depth_map=grad_d(p,q,xcoords)
        # warps.append(tf.squeeze(tf.slice(scaled_images,[0,0,0,0,0],[-1,1,-1,-1,-1]),1))
        # for view in range(1,FLAGS.view_num):
        #     view_cam=tf.squeeze(tf.slice(scaled_cams,[0,view,0,0,0],[-1,1,-1,-1,-1]),1)
        #     view_image=tf.squeeze(tf.slice(scaled_images,[0,view,0,0,0],[-1,1,-1,-1,-1]),1)
        #     warped,_,_=reprojection_depth(view_image,ref_cam,view_cam,init_depth_map)
        #     warps.append(warped)
        # warps=tf.stack(warps,-1)#b,h,w,3,1
        # warps=tf.reduce_mean(warps**2,axis=-1)-tf.reduce_mean(warps,axis=-1)**2
        # sigma=tf.reduce_mean(warps,axis=-1,keepdims=True)#b,h,w,1
        # mask=tf.cast(sigma<25.0,tf.float32)
        # mask=tf.reshape(mask,[batch_size,height,width,1])
        # init_depth_map=init_depth_map*mask
        # view_img=tf.squeeze(tf.slice(scaled_images,[0,2,0,0,0],[-1,1,-1,-1,-1]),1)
        # # depth_map=init_depth_map*tf.cast(prob_map>0.6,tf.float32)
        # warped,_=reprojection(view_img,ref_cam,view_cam,init_depth_map)

        # x_linspace = tf.linspace(0., tf.cast(width, 'float32'), width)
        # y_linspace = tf.linspace(0., tf.cast(height, 'float32'), height)
        # x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        # x_coordinates=tf.tile(tf.reshape(x_coordinates,[1,height,width,1]),[batch_size,1,1,1])
        # xcoords=x_coordinates+flow

        # x_linspace = tf.linspace(0., tf.cast(width, 'float32'), width)
        # y_linspace = tf.linspace(0., tf.cast(height, 'float32'), height)
        # x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        # x_coordinates=tf.tile(tf.reshape(x_coordinates,[1,height,width,1]),[batch_size,1,1,1])
        # opt=tf.train.AdamOptimizer(1e-2)
        # # init_depth_map=init_depth_map*tf.cast(prob_map>0.3,tf.float32)

        # # print(tf.shape(init_depth_map))
        # # depth_map=tf.Variable(tf.zeros_like(init_depth_map),trainable=True,name='refine_depth_map')
        # flows=tf.stack(flows,1)
        # with tf.variable_scope('optim',reuse=tf.AUTO_REUSE):
        #     depth_map = tf.get_variable("refine_depth_map", dtype=tf.float32,initializer=tf.zeros([batch_size,height,width,1],tf.float32))
        # # # tf.assign(depth_map,init_depth_map)
        # # # depth_map=tf.assign(depth_map,init_depth_map)
        # cams=tf.placeholder(shape=[None,FLAGS.view_num,2,4,4],dtype=tf.float32,name='cams')
        # ref_cam=tf.squeeze(tf.slice(cams,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
        # coords=[]
        # gt_coords=[]
        # for view in range(1,FLAGS.view_num):
        #     flow=tf.squeeze(tf.slice(flows,[0,view-1,0,0,0],[-1,1,-1,-1,1]),1)#b,h,w,1
        #     view_cam=tf.squeeze(tf.slice(cams,[0,view,0,0,0],[-1,1,-1,-1,-1]),1)
        #     coord=reprojection_point([batch_size,height,width],ref_cam,view_cam,depth_map)#b,h,w,2
        #     coord=coord*tf.cast( (coord>0 )&( coord<width),tf.float32)
        #     coord=tf.reshape(coord,[batch_size,height,width,1])
        #     gt_coord=tf.stop_gradient(x_coordinates+flow)
        #     coords.append(coord)
        #     gt_coords.append(gt_coord)
        # coords=tf.concat(coords,-1)#b,h,w,1,3
        # gt_coords=tf.concat(gt_coords,-1)
        # loss1=tf.reduce_mean(tf.square(gt_coords-coords),axis=-1,keepdims=True)
        # flow11=tf.placeholder(shape=[None,None,None,1],dtype=tf.float32,name='flow1')
        # flow12=tf.placeholder(shape=[None,None,None,1],dtype=tf.float32,name='flow2')

        # ref_cam=tf.squeeze(tf.slice(cams,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)

        # view1_cam=tf.squeeze(tf.slice(cams,[0,1,0,0,0],[-1,1,-1,-1,-1]),1)
        # view2_cam=tf.squeeze(tf.slice(cams,[0,2,0,0,0],[-1,1,-1,-1,-1]),1)
        # ref_cam=tf.stop_gradient(ref_cam)
        # view1_cam=tf.stop_gradient(view1_cam)
        # view2_cam=tf.stop_gradient(view2_cam)

        # # global_coords=reprojection_global_point([batch_size,height,width],ref_cam,view1_cam,depth_map)
        # coords1=reprojection_point([batch_size,height,width],ref_cam,view1_cam,depth_map)#b,h,w,2
        # coords1=coords1*tf.cast( (coords1>0 )&( coords1<width),tf.float32)
        # coords1=tf.reshape(coords1,[batch_size,height,width,1])

        # coords2=reprojection_point([batch_size,height,width],ref_cam,view2_cam,depth_map)
        # coords2=coords2*tf.cast( (coords2>0 )&( coords2<width),tf.float32)
        # coords2=tf.reshape(coords2,[batch_size,height,width,1])
        # coords=tf.concat([coords1,coords2],-1)
        # gt_coords_11=tf.stop_gradient(x_coordinates+flow11)
        # gt_coords_12=tf.stop_gradient(x_coordinates+flow12)
        # gt_coords=tf.concat([gt_coords_11,gt_coords_12],-1)
        # gt_coords=tf.reshape(gt_coords,[batch_size,height,width,2])
        # gt_coords=gt_coords*tf.cast( (gt_coords>0 )&( gt_coords<width),tf.float32)

        # loss1=tf.reduce_mean(tf.square(gt_coords-coords),axis=-1,keepdims=True)
        # mask1=tf.stop_gradient(tf.cast((loss1<10.0),tf.float32))
        # valid_count=tf.count_nonzero(mask1,dtype=tf.float32)*2.0+1e-8
        # residual=tf.reduce_mean(tf.square(gt_coords-coords),axis=-1,keepdims=True)*mask1
        # loss=tf.reduce_sum(residual)/valid_count



        # # loss=tf.nn.l2_loss(gt_coords-coords)
        # grads=opt.compute_gradients(loss)
        # # print(grads)
        # grads=[grad for grad in grads if grad[0] is not None]
        # train_opts=opt.apply_gradients(grads)
        # p,q=PQ(ref_cam,view_cam,[batch_size,height,width,1])
        # depth=grad_d(p,q,xcoords)


        # with tf.name_scope("Model_tower0"):
        #     prob_volume = inference_1(
        #             centered_images, scaled_cams, FLAGS.max_d, depth_start, depth_end,True)
#             depth_end=depth_interval*(FLAGS.max_d-1)+depth_start
        # prob_map=tf.reduce_max(prob_volume,axis=1)#b,h,w,1
        # prob_index=tf.cast(tf.argmax(prob_volume,axis=1),tf.float32)#b,h,w,1
        # # if FLAGS.inverse_depth:
        # inv_depth_start = tf.div(1.0, depth_start)
        # inv_depth_end = tf.div(1.0, depth_end)
        # inv_interval = (inv_depth_start - inv_depth_end) / (FLAGS.max_d - 1)
        # inv_depth = inv_depth_start - prob_index * inv_interval
        # init_depth_map = tf.div(1.0, inv_depth)
        # prob_map = get_propability_map(prob_volume, inv_depth, inv_depth_start, -inv_interval)

    # init option
    init_op = tf.global_variables_initializer()
    var_init_op = tf.local_variables_initializer()





    # initialization
    sess.run(var_init_op)
    sess.run(init_op)
    total_step = 0


    # load model
    if FLAGS.model_dir is not None:
        # pretrained_model_ckpt_path = os.path.join(FLAGS.model_dir,"19-09-16-10", FLAGS.regularization, 'model.ckpt')
        # pretrained_model_ckpt_path = os.path.join("/home/haibao637/data/tf_models/","unet1_uconvlstm","GRU", 'model.ckpt')

        "snetgn_singleHomo_uconv_lstm_128_both_32_fix_interval_scale_1.0"

        pretrained_model_ckpt_path = os.path.join("/home/haibao637/data/tf_models/","snetgn_singlhomo_ulstm_128_both_interval_scale_1.08_local_global","GRU", 'model.ckpt')

        restorer = tf.train.Saver(tf.global_variables())
        pretrained_model_ckpt_path='-'.join([pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])
        restorer.restore(sess,pretrained_model_ckpt_path )
        # optimistic_restore(sess,pretrained_model_ckpt_path)
        print(Notify.INFO, 'Pre-trained model restored from %s' %
                (pretrained_model_ckpt_path), Notify.ENDC)
        total_step = FLAGS.ckpt_step

    # run inference for each reference view3
    sess.run(mvs_iterator.initializer)
    for step in range(len(mvs_list)):

        start_time = time.time()
        try:
            out_init_depth_map,out_prob_map, out_images, out_cams, out_index = sess.run(
                [init_depth_map, prob_map,scaled_images, scaled_cams, image_index])
            # print(out_index)

            out_init_depth_map=np.clip(out_init_depth_map,0.0,10.0)
            # out_init_depth_map[out_prob_map<0.3]=0.0



            # cam_params=np.zeros([9])
            # cam_params[:3] = R.from_dcm(out_cams[0,0,:3,:3])
            # cam_params[3:6]=out_cams[0,0,:3,3]
            # cam_params[6]=out_cams[0,1,0,0]
            # cam_params[7:]=out_cams[0,1,:2,2]

            # depth_map.load(out_init_depth_map,sess)
            # # for i in range(100):
            # #     depth_map.load(out_init_depth_map,sess)
            # #     out_loss,out_opts,out_grad=sess.run([loss,train_opts,grads],
            # #         feed_dict={
            # #             flow11:out_flow1,
            # #             flow12:out_flow2,
            # #             cams:out_cams
            # #         }
            # #     )

            # #     out_init_depth_map=sess.run(depth_map)
            # #     out_init_depth_map=np.clip(out_init_depth_map,0,10)

            # depth_map.load(out_init_depth_map,sess)
            # out_loss=sess.run(loss1,feed_dict={
            #             flows:out_flows,

            #             cams:out_cams,

            #         }
            # )
            # print("before optimizer:",np.count_nonzero(out_loss<1.0))
            # # out_init_depth_map[out_loss>0.1]=0.0
            # # depth_map.load(out_init_depth_map,sess)
            # # optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 50})

            # # optimizer.minimize(sess,feed_dict={
            # #             flows:out_flows,
            # #             cams:out_cams,
            # #             loss1:out_loss,

            # #         })
            # # # out_loss,out_opts,out_grad=sess.run([loss,train_opts,grads],
            # # #         feed_dict={
            # # #             flow11:out_flow1,
            # # #             flow12:out_flow2,
            # # #             cams:out_cams
            # # #         }
            # # #     )
            # # # out_init_depth_map=sess.run(depth_map)
            # # out_init_depth_map,out_loss=sess.run([depth_map,loss1],feed_dict={
            # #             flows:out_flows,
            # #             cams:out_cams,

            # #         })

            # # out_init_depth_map=sess.run(depth_map)
            # # print(out_loss.shape,out_init_depth_map.shape)
            # # out_init_depth_map=np.clip(out_init_depth_map,0,10)
            # out_init_depth_map[out_loss>1.0]=0.0
            # # out_init_depth_map=np.clip(out_init_depth_map,0,10)
            # # out_init_depth_map[out_loss>1.0]=0.0
            # print("after optimizer:",np.count_nonzero(out_loss<1.0))
        except tf.errors.OutOfRangeError:
            print("all dense finished")  # ==> "End of dataset"
            break
        duration = time.time() - start_time
        print(Notify.INFO, 'depth inference %d finished. (%.3f sec/step)' % (step, duration),
                Notify.ENDC)

        # squeeze output
        out_init_depth_image = np.squeeze(out_init_depth_map)
        out_prob_map = np.squeeze(out_prob_map)
        out_ref_image = np.squeeze(out_images)
        out_ref_image = np.squeeze(out_ref_image[0, :, :, :])
        out_ref_cam = np.squeeze(out_cams)
        out_ref_cam = np.squeeze(out_ref_cam[0, :, :, :])
        out_index = np.squeeze(out_index)
        # out_warped=np.squeeze(out_warped,0)
        # paths
        init_depth_map_path = output_folder + ('/%08d_init.pfm' % out_index)
        prob_map_path = output_folder + ('/%08d_prob.pfm' % out_index)
        out_ref_image_path = output_folder + ('/%08d.jpg' % out_index)
        out_ref_cam_path = output_folder + ('/%08d.txt' % out_index)
        # scipy.misc.imsave(os.path.join(output_folder,'%08d_warped.jpg'%out_index),out_warped)
        # save output
        write_pfm(init_depth_map_path, out_init_depth_image)
        print(out_prob_map.shape)
        out_prob_map=cv2.resize(out_prob_map,(out_init_depth_image.shape[1],out_init_depth_image.shape[0]))
        write_pfm(prob_map_path, out_prob_map)
        # out_ref_image = cv2.cvtColor(out_ref_image, cv2.COLOR_RGB2BGR)
        out_ref_image=cv2.resize(out_ref_image,(out_init_depth_image.shape[1],out_init_depth_image.shape[0]))
        # image_file = file_io.FileIO(out_ref_image_path, mode='w')
        cv2.imwrite(out_ref_image_path, out_ref_image)
        write_cam(out_ref_cam_path, out_ref_cam)
        total_step += 1


def main(_):  # pylint: disable=unused-argument
    """ program entrance """
    # generate input path list
    mvs_list = gen_pipeline_mvs_list(FLAGS.dense_folder)
    # print([[item.image,item.dis] for item in mvs_list[32]])
    # mvsnet inference
    mvsnet_pipeline(mvs_list)


if __name__ == '__main__':
    tf.app.run()
