#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Differentiable homography related.
"""

import tensorflow as tf
import numpy as np
def get_homography(left_cam,right_cam,depth,scale):
    R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
    R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
    t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
    t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
    K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
    scales = tf.constant([scale, scale, 1])
    K_left = K_left * scales
    K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
    K_right = K_right * scales
    # depth

    # depth_num = tf.reshape(tf.cast(depth_num, 'int32'), [])
    # depth = depth_start + tf.cast(tf.range(depth_num), tf.float32) * depth_interval
    # preparation
    num_depth = tf.shape(depth)[0]
    K_left_inv = tf.matrix_inverse(tf.squeeze(K_left, axis=1))
    R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
    R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])

    fronto_direction = tf.slice(tf.squeeze(R_left, axis=1), [0, 2, 0], [-1, 1, 3])  # (B, D, 1, 3)

    c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
    c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))  # (B, D, 3, 1)
    c_relative = tf.subtract(c_right, c_left)

    # compute
    batch_size = tf.shape(R_left)[0]
    temp_vec = tf.matmul(c_relative, fronto_direction)
    depth_mat = tf.tile(tf.reshape(depth, [batch_size, num_depth, 1, 1]), [1, 1, 3, 3])

    temp_vec = tf.tile(tf.expand_dims(temp_vec, axis=1), [1, num_depth, 1, 1])

    middle_mat0 = tf.eye(3, batch_shape=[batch_size, num_depth]) - temp_vec / depth_mat
    middle_mat1 = tf.tile(tf.expand_dims(tf.matmul(R_left_trans, K_left_inv), axis=1), [1, num_depth, 1, 1])
    middle_mat2 = tf.matmul(middle_mat0, middle_mat1)

    homographies = tf.matmul(tf.tile(K_right, [1, num_depth, 1, 1])
                             , tf.matmul(tf.tile(R_right, [1, num_depth, 1, 1])
                                         , middle_mat2))

    return homographies#b,1,3,3
def PQ(left_cam,right_cam,image_shape):
    batch_size = image_shape[0]
    height = image_shape[1]
    width = image_shape[2]
    pixel_grids = get_pixel_grids(height, width)
    pixel_grids = tf.expand_dims(pixel_grids, 0)
    pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
    pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))  # b,3,[hxw]

    R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
    R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
    t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
    t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])#b,1,3,1
    K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
    K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
    K_left_inv = tf.squeeze(tf.matrix_inverse(K_left), 1)
    # K_right=tf.squeeze(K_right,1)
    R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
    R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])
    c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
    c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))  # (B, D, 3, 1)
    c_relative = tf.subtract(c_right, c_left)  # b,3,1
    P = tf.matmul(tf.squeeze(tf.matmul(K_right, R_right),1), tf.matmul(R_left_trans, K_left_inv))
    Q = tf.matmul(tf.squeeze(tf.matmul(K_right, R_right),1), c_relative)
    P = tf.matmul(P, pixel_grids)#b,3,[h*w]
    return P, Q
def grad_d(P,Q,x,axis=0):
    shape=tf.shape(x)
    batch_size=shape[0]
    height=shape[1]
    width=shape[2]

    x=tf.reshape(x,[batch_size,1,-1])
    p0=tf.slice(P,[0,axis,0],[-1,1,-1])
    p1=tf.slice(P,[0,1,0],[-1,1,-1])
    p2=tf.slice(P,[0,2,0],[-1,1,-1])
    q0=tf.slice(Q,[0,axis,0],[-1,1,-1])
    q1=tf.slice(Q,[0,1,0],[-1,1,-1])
    q2=tf.slice(Q,[0,2,0],[-1,1,-1])
    up=q2*x-q0
    div=p2*x-p0
    
    mask=tf.cast(div==0.0,dtype=tf.float32)
    div=mask*div+1e-7+(1-mask)*div
    depth=tf.divide(up,div)
    depth=depth*(1-mask)
    y_coords=p1*depth-q1
    view_d=p2*depth-q2
    mask=tf.cast((view_d<=0),tf.float32)
    depth=depth*(1-mask)
    view_d=view_d*(1-mask)+mask*1e-7
    y_coords=tf.divide(y_coords,view_d)
    mask=tf.cast((y_coords>=0)&(y_coords<tf.cast(height,tf.float32))&(depth>0),tf.float32)
    depth=depth*mask
    depth=tf.reshape(depth,[batch_size,height,width,1])
    return depth
def projection_grid(left_cam,points):
    """

    cam:b,2,4,4
    points:b,3,dhw
    """
    """
     cam*points ->b,dhw,3
     interprate -> b,d,h,w,c
    """
    
    R_left = tf.squeeze(tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3]),1)
   
    t_left = tf.squeeze(tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1]),1)
  
    K_left = tf.squeeze(tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3]),1)
    R_left_trans = tf.transpose(R_left, perm=[0, 2, 1])
  
    c_left = -tf.matmul(R_left_trans, t_left)
    middle_mat1 = tf.matmul(K_left, R_left)#b,3,3
    points=tf.subtract(points,c_left)
    grids=tf.matmul(middle_mat1,points-c_left)#b,3,dhw
    div=tf.slice(grids,[0,2,0],[-1,1,-1])
    up=tf.slice(grids,[0,0,0],[-1,2,-1])
    mask=tf.cast(div<0,tf.float32)
    div=(1-mask)*div+mask*1e-5
    # coords=tf.divide(up.div)
    coords=tf.divide(up,div)#b,2,dhw
    return coords
    
def global_points(left_cam,depth):
    """
    lef_cam:b,2,4,4
    depth:b,h,w,1
    return: b,3,hxw
    """
    image_shape=tf.shape(depth)
    batch_size = image_shape[0]
    height = image_shape[1]
    width = image_shape[2]
    pixel_grids = get_pixel_grids(height, width)
    pixel_grids = tf.expand_dims(pixel_grids, 0)
    pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
    pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))  # b,3,[hxw]

    R_left = tf.squeeze(tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3]),1)
   
    t_left = tf.squeeze(tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1]),1)
  
    K_left = tf.squeeze(tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3]),1)
   
    K_left_inv =tf.matrix_inverse(K_left)
    # K_right=tf.squeeze(K_right,1)
    R_left_trans = tf.transpose(R_left, perm=[0, 2, 1])
  
    c_left = -tf.matmul(R_left_trans, t_left)
    pixel_grids = get_pixel_grids(height, width)
    pixel_grids = tf.expand_dims(pixel_grids, 0)
    pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
    pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))#b,3,[hxw]
    depth_flatten=tf.reshape(depth,(batch_size,1,-1))#b,1,[hxw]
    pixel_grids=pixel_grids*depth_flatten
    middle_mat1 = tf.matmul(R_left_trans, K_left_inv)#b,3,3
    pixel_grids=tf.matmul(middle_mat1,pixel_grids)#b,3,[h,w]
    pixel_grids=tf.add(pixel_grids,c_left)
    return pixel_grids



def grad_dy(P,Q,x,axis=0):
    shape=tf.shape(x)
    batch_size=shape[0]
    height=shape[1]
    width=shape[2]

    x=tf.reshape(x,[batch_size,1,-1])
    p0=tf.slice(P,[0,axis,0],[-1,1,-1])
    p1=tf.slice(P,[0,1,0],[-1,1,-1])
    p2=tf.slice(P,[0,2,0],[-1,1,-1])
    q0=tf.slice(Q,[0,axis,0],[-1,1,-1])
    q1=tf.slice(Q,[0,1,0],[-1,1,-1])
    q2=tf.slice(Q,[0,2,0],[-1,1,-1])
    up=q2*x-q0
    div=p2*x-p0
    
    mask=tf.cast(div==0.0,dtype=tf.float32)
    div=mask*div+1e-7+(1-mask)*div
    depth=tf.divide(up,div)
    y_coords=p1*depth-q1
    view_d=p2*depth-q2
    mask=tf.cast((view_d<=0),tf.float32)
    depth=depth*(1-mask)
    view_d=view_d*(1-mask)+mask*1e-7
    y_coords=tf.divide(y_coords,view_d)
    mask=tf.cast((y_coords>=0)&(y_coords<tf.cast(height,tf.float32)),tf.float32)
    depth=depth*mask
    depth=tf.reshape(depth,[batch_size,height,width,1])
    return depth


def get_homographies(left_cam, right_cam, depth_num, depth_start, depth_interval,scale=1.0):
    with tf.name_scope('get_homographies'):
        # cameras (K, R, t)
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        scales=tf.reshape(tf.constant([scale,scale,1]),[1,1,3])
        K_left=K_left*scales
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right=K_right*scales
        # depth

        depth_num = tf.reshape(tf.cast(depth_num, 'int32'), [])
        depth = depth_start + tf.cast(tf.range(depth_num), tf.float32) * depth_interval
        # preparation
        num_depth = depth_num
        K_left_inv = tf.matrix_inverse(tf.squeeze(K_left, axis=1))
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])

        fronto_direction = tf.slice(tf.squeeze(R_left, axis=1), [0, 2, 0], [-1, 1, 3])          # (B, D, 1, 3)

        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))                        # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)        

        # compute
        batch_size = tf.shape(R_left)[0]
        temp_vec = tf.matmul(c_relative, fronto_direction)
        depth_mat = tf.tile(tf.reshape(depth, [batch_size, num_depth, 1, 1]), [1, 1, 3, 3])

        temp_vec = tf.tile(tf.expand_dims(temp_vec, axis=1), [1, num_depth, 1, 1])

        middle_mat0 = tf.eye(3, batch_shape=[batch_size, num_depth]) - temp_vec / depth_mat
        middle_mat1 = tf.tile(tf.expand_dims(tf.matmul(R_left_trans, K_left_inv), axis=1), [1, num_depth, 1, 1])
        middle_mat2 = tf.matmul(middle_mat0, middle_mat1)

        homographies = tf.matmul(tf.tile(K_right, [1, num_depth, 1, 1])
                     , tf.matmul(tf.tile(R_right, [1, num_depth, 1, 1])
                     , middle_mat2))

    return homographies

def reprojection_global_point(image_shape,left_cam,depth_map):
    """

    :param input_image:
    :param left_cam:
    :param right_cam:
    :param depth_map: b,h,w,1
    :return:
    """
    with tf.name_scope('warping_by_homography'):
       
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]
        pixel_grids = get_pixel_grids(height, width)
        pixel_grids = tf.expand_dims(pixel_grids, 0)
        pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
        pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))#b,3,[hxw]
        depth_flatten=tf.reshape(depth_map,(batch_size,1,-1))#b,1,[hxw]
        pixel_grids=pixel_grids*depth_flatten
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])

        K_left_inv = tf.squeeze(tf.matrix_inverse(K_left),1)
        # K_right=tf.squeeze(K_right,1)
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        
        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
 
        c_relative = c_left
        middle_mat1 = tf.matmul(R_left_trans, K_left_inv)#b,3,3
        pixel_grids=tf.matmul(middle_mat1,pixel_grids)#b,3,[h,w]
        pixel_grids=tf.subtract(pixel_grids,c_relative)
        reprojected_point=tf.transpose(tf.reshape(pixel_grids,[batch_size,3,height,width]),[0,2,3,1])
        return reprojected_point
def reprojection_point(image_shape,left_cam,right_cam,depth_map):
    """

    :param input_image:
    :param left_cam:
    :param right_cam:
    :param depth_map: b,h,w,1
    :return:
    """
    with tf.name_scope('warping_by_homography'):
       
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]
        pixel_grids = get_pixel_grids(height, width)
        pixel_grids = tf.expand_dims(pixel_grids, 0)
        pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
        pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))#b,3,[hxw]
        depth_flatten=tf.reshape(depth_map,(batch_size,1,-1))#b,1,[hxw]
        pixel_grids=pixel_grids*depth_flatten
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_left_inv = tf.squeeze(tf.matrix_inverse(K_left),1)
        # K_right=tf.squeeze(K_right,1)
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])
        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))  # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)#b,3,1
        middle_mat1 = tf.matmul(R_left_trans, K_left_inv)#b,3,3
        middle_mat2 = tf.squeeze(tf.matmul(K_right, R_right),1)  # b,3,3
        pixel_grids=tf.matmul(middle_mat1,pixel_grids)#b,3,[h,w]
        pixel_grids=tf.subtract(pixel_grids,c_relative)
        pixel_grids=tf.matmul(middle_mat2,pixel_grids)
        grids_div=tf.slice(pixel_grids,[0,2,0],[-1,1,-1])#b,1,[h,w]
        grids_zero_add = tf.cast(tf.equal(grids_div, 0.0), dtype='float32') * 1e-7  # handle div 0
        grids_div = grids_div + grids_zero_add
        grids_div = tf.tile(grids_div, [1, 2, 1])
        grids_affine=tf.slice(pixel_grids,[0,0,0],[-1,2,-1])
        grids_inv_warped = tf.div(grids_affine, grids_div)
        reprojected_point=tf.transpose(tf.reshape(grids_inv_warped,[batch_size,2,height,width]),[0,2,3,1])
        return tf.slice(reprojected_point,[0,0,0,0],[-1,-1,-1,1])
    
        

def reprojection_depth(input_image,left_cam,right_cam,depth_map):
    """

    :param input_image:
    :param left_cam:
    :param right_cam:
    :param depth_map: b,h,w,1
    :return:
    """
    with tf.name_scope('warping_by_homography'):
        image_shape = tf.shape(input_image)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]
        pixel_grids = get_pixel_grids(height, width)
        pixel_grids = tf.expand_dims(pixel_grids, 0)
        pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
        pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))#b,3,[hxw]
        depth_flatten=tf.reshape(depth_map,(batch_size,1,-1))#b,1,[hxw]
        pixel_grids=pixel_grids*depth_flatten
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_left_inv = tf.squeeze(tf.matrix_inverse(K_left),1)
        # K_right=tf.squeeze(K_right,1)
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])
        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))  # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)#b,3,1
        middle_mat1 = tf.matmul(R_left_trans, K_left_inv)#b,3,3
        middle_mat2 = tf.squeeze(tf.matmul(K_right, R_right),1)  # b,3,3
        pixel_grids=tf.matmul(middle_mat1,pixel_grids)#b,3,[h,w]
        pixel_grids=tf.subtract(pixel_grids,c_relative)
        pixel_grids=tf.matmul(middle_mat2,pixel_grids)
        grid_div=tf.slice(pixel_grids,[0,2,0],[-1,1,-1])#b,1,[h,w]
        grid_zero_add = tf.cast(tf.equal(grid_div, 0.0), dtype='float32') * 1e-7  # handle div 0
        grid_div = grid_div + grid_zero_add
        grids_div = tf.tile(grid_div, [1, 2, 1])
        grids_affine=tf.slice(pixel_grids,[0,0,0],[-1,2,-1])
        grids_inv_warped = tf.div(grids_affine, grids_div)
        x_warped, y_warped = tf.unstack(grids_inv_warped, axis=1)
        x_warped_flatten = tf.reshape(x_warped, [-1])
        y_warped_flatten = tf.reshape(y_warped, [-1])
        grid_div=tf.reshape(grid_div,tf.shape(x_warped))
        warped_image = interpolate(input_image, x_warped_flatten, y_warped_flatten)
        # warped_image = tf.reshape(input_image, shape=image_shape, name='warped_feature')
        mask=tf.reshape(tf.greater(grid_div, 0.0)&(x_warped>=0.0) & (x_warped<tf.cast(width,tf.float32)) & (y_warped>=0.0) &( y_warped<tf.cast(height,tf.float32)),
                        [batch_size,height,width,1])
        grid_depth = tf.reshape(grid_div, [batch_size,height,width,1])
        return warped_image,grid_depth,mask
    # """

    # :param input_image:
    # :param left_cam:
    # :param right_cam:
    # :param depth_map: b,h,w,1
    # :return:
    # """
    # with tf.name_scope('warping_by_homography'):
    #     image_shape = tf.shape(input_image)
    #     batch_size = image_shape[0]
    #     height = image_shape[1]
    #     width = image_shape[2]
    #     pixel_grids = get_pixel_grids(height, width)
    #     pixel_grids = tf.expand_dims(pixel_grids, 0)
    #     pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
    #     pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))#b,3,[hxw]
    #     depth_flatten=tf.reshape(depth_map,(batch_size,1,-1))#b,1,[hxw]
    #     pixel_grids=pixel_grids*depth_flatten
    #     R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
    #     R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
    #     t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
    #     t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
    #     K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
    #     K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
    #     K_left_inv = tf.squeeze(tf.matrix_inverse(K_left),1)
    #     # K_right=tf.squeeze(K_right,1)
    #     R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
    #     R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])
    #     c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
    #     c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))  # (B, D, 3, 1)
    #     c_relative = tf.subtract(c_right, c_left)#b,3,1
    #     middle_mat1 = tf.matmul(R_left_trans, K_left_inv)#b,3,3
    #     middle_mat2 = tf.squeeze(tf.matmul(K_right, R_right),1)  # b,3,3
    #     pixel_grids=tf.matmul(middle_mat1,pixel_grids)#b,3,[h,w]
    #     pixel_grids=tf.subtract(pixel_grids,c_relative)
    #     pixel_grids=tf.matmul(middle_mat2,pixel_grids)
    #     grids_div=tf.slice(pixel_grids,[0,2,0],[-1,1,-1])#b,1,[h,w]
    #     grids_zero_add = tf.cast(tf.equal(grids_div, 0.0), dtype='float32') * 1e-7  # handle div 0
    #     # grids_div = grids_div + grids_zero_add
    #     grids_divs = tf.tile(grids_div+ grids_zero_add, [1, 2, 1])
    #     grids_affine=tf.slice(pixel_grids,[0,0,0],[-1,2,-1])
    #     grids_inv_warped = tf.div(grids_affine, grids_divs)
    #     x_warped, y_warped = tf.unstack(grids_inv_warped, axis=1)
    #     # x_warped=tf.reshape(x_warped,[batch_size,height,width,1])
    #     x_warped_flatten = tf.reshape(x_warped, [-1])
    #     y_warped_flatten = tf.reshape(y_warped, [-1])
    #     grids_div=tf.reshape(grids_div,tf.shape(x_warped))
    #     warped_image = interpolate(input_image, x_warped_flatten, y_warped_flatten)
    #     warped_image = tf.reshape(warped_image, shape=image_shape, name='warped_feature')
    #     mask=tf.reshape(tf.greater(grids_div, 0.0)&(x_warped>=0.0) & (x_warped<tf.cast(width,tf.float32)) & (y_warped>=0.0) &( y_warped<tf.cast(height,tf.float32)),
    #                     [batch_size,height,width,1])
    #     grids_depth = tf.reshape(grids_div, [batch_size,height,width,1])
    #     return warped_image,grids_depth,mask
def get_homographies_inv_depth(left_cam, right_cam, depth_num, depth_start, depth_end,scale=1.0):

    with tf.name_scope('get_homographies'):
        # cameras (K, R, t)
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        # scales = tf.constant([scale, scale, 1])
        scales=tf.reshape(tf.constant([scale,scale,1]),[1,1,3])
        K_left = K_left * scales
        K_right = K_right * scales
        # depth
        # if depth is None:
        depth_num = tf.reshape(tf.cast(depth_num, 'int32'), [])

        inv_depth_start = tf.reshape(tf.div(1.0, depth_start), [])
        inv_depth_end = tf.reshape(tf.div(1.0, depth_end), [])
        inv_depth = tf.lin_space(inv_depth_start, inv_depth_end, depth_num)
        depth = tf.div(1.0, inv_depth)
        # else:
        #     depth=tf.div(1.0,depth)

        # preparation
        num_depth = tf.shape(depth)[0]
        K_left_inv = tf.matrix_inverse(tf.squeeze(K_left, axis=1))
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])

        fronto_direction = tf.slice(tf.squeeze(R_left, axis=1), [0, 2, 0], [-1, 1, 3])          # (B, D, 1, 3)

        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))                        # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)

        # compute
        batch_size = tf.shape(R_left)[0]
        temp_vec = tf.matmul(c_relative, fronto_direction)
        depth_mat = tf.tile(tf.reshape(depth, [batch_size, num_depth, 1, 1]), [1, 1, 3, 3])

        temp_vec = tf.tile(tf.expand_dims(temp_vec, axis=1), [1, num_depth, 1, 1])

        middle_mat0 = tf.eye(3, batch_shape=[batch_size, num_depth]) - temp_vec / depth_mat
        middle_mat1 = tf.tile(tf.expand_dims(tf.matmul(R_left_trans, K_left_inv), axis=1), [1, num_depth, 1, 1])
        middle_mat2 = tf.matmul(middle_mat0, middle_mat1)

        homographies = tf.matmul(tf.tile(K_right, [1, num_depth, 1, 1])
                     , tf.matmul(tf.tile(R_right, [1, num_depth, 1, 1])
                     , middle_mat2))

    return homographies
def get_homographies_inv_depth_1(left_cam, right_cam, depth_num, depth_start, depth_end):

    with tf.name_scope('get_homographies'):
        # cameras (K, R, t)
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])

        # depth
        depth_num = tf.reshape(tf.cast(depth_num, 'int32'), [])

        inv_depth_start = tf.reshape(tf.div(1.0, depth_start), [])
        inv_depth_end = tf.reshape(tf.div(1.0, depth_end), [])
        inv_depth = tf.lin_space(inv_depth_start, inv_depth_end, depth_num)
        depth = tf.div(1.0, inv_depth)

        # preparation
        num_depth = tf.shape(depth)[0]
        K_left_inv = tf.matrix_inverse(tf.squeeze(K_left, axis=1))
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])

        fronto_direction = tf.slice(tf.squeeze(R_left, axis=1), [0, 2, 0], [-1, 1, 3])          # (B, D, 1, 3)

        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))                        # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)

        # compute
        batch_size = tf.shape(R_left)[0]
        temp_vec = tf.matmul(c_relative, fronto_direction)
        depth_mat = tf.tile(tf.reshape(depth, [batch_size, num_depth, 1, 1]), [1, 1, 3, 3])

        temp_vec = tf.tile(tf.expand_dims(temp_vec, axis=1), [1, num_depth, 1, 1])

        middle_mat0 = tf.eye(3, batch_shape=[batch_size, num_depth]) - temp_vec / depth_mat
        middle_mat1 = tf.tile(tf.expand_dims(tf.matmul(R_left_trans, K_left_inv), axis=1), [1, num_depth, 1, 1])
        middle_mat2 = tf.matmul(middle_mat0, middle_mat1)

        homographies = tf.matmul(tf.tile(K_right, [1, num_depth, 1, 1])
                     , tf.matmul(tf.tile(R_right, [1, num_depth, 1, 1])
                     , middle_mat2))

    return homographies
def get_pixel_grids(height, width):
    # texture coordinate
    x_linspace = tf.linspace(0., tf.cast(width, 'float32') - 0., width)
    y_linspace = tf.linspace(0., tf.cast(height, 'float32') - 0., height)
    x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
    x_coordinates = tf.reshape(x_coordinates, [-1])
    y_coordinates = tf.reshape(y_coordinates, [-1])
    ones = tf.ones_like(x_coordinates)
    indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)#[-1]
    return indices_grid

def repeat_int(x, num_repeats):
    ones = tf.ones((1, num_repeats), dtype='int32')
    x = tf.reshape(x, shape=(-1, 1))
    x = tf.matmul(x, ones)
    return tf.reshape(x, [-1])

def repeat_float(x, num_repeats):
    ones = tf.ones((1, num_repeats), dtype='float')
    x = tf.reshape(x, shape=(-1, 1))
    x = tf.matmul(x, ones)
    return tf.reshape(x, [-1])

def interpolate(image, x, y):
    image_shape = tf.shape(image)
    batch_size = image_shape[0]
    height =image_shape[1]
    width = image_shape[2]

    # image coordinate to pixel coordinate
    x = x - 0.5
    y = y - 0.5
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    max_y = tf.cast(height - 1, dtype='int32')
    max_x = tf.cast(width - 1,  dtype='int32')
    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)
    b = repeat_int(tf.range(batch_size), height * width)

    indices_a = tf.stack([b, y0, x0], axis=1)
    indices_b = tf.stack([b, y0, x1], axis=1)
    indices_c = tf.stack([b, y1, x0], axis=1)
    indices_d = tf.stack([b, y1, x1], axis=1)

    pixel_values_a = tf.gather_nd(image, indices_a)
    pixel_values_b = tf.gather_nd(image, indices_b)
    pixel_values_c = tf.gather_nd(image, indices_c)
    pixel_values_d = tf.gather_nd(image, indices_d)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    area_a = tf.expand_dims(((y1 - y) * (x1 - x)), 1)
    area_b = tf.expand_dims(((y1 - y) * (x - x0)), 1)
    area_c = tf.expand_dims(((y - y0) * (x1 - x)), 1)
    area_d = tf.expand_dims(((y - y0) * (x - x0)), 1)
    # total=(area_a+area_b+area_c+area_d)
    output = tf.add_n([area_a * pixel_values_a,
                        area_b * pixel_values_b,
                        area_c * pixel_values_c,
                        area_d * pixel_values_d])
    # mask=tf.cast(total>0,tf.float32)
    # total=mask*total+(1-mask)
    # output=output/total
    
    return tf.reshape(output,tf.shape(image))
def interpolate_grid(image, x, y,pad_size=9):
    image_shape = tf.shape(image)
    batch_size = image_shape[0]
    height =image_shape[1]
    width = image_shape[2]
    channels=image_shape[3]
    # image coordinate to pixel coordinate
    x = x - 0.5
    y = y - 0.5
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    max_y = tf.cast(height - 1, dtype='int32')
    max_x = tf.cast(width - 1,  dtype='int32')
    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)
    b = repeat_int(tf.range(batch_size), height * width*pad_size)

    indices_a = tf.stack([b, y0, x0], axis=1)
    indices_b = tf.stack([b, y0, x1], axis=1)
    indices_c = tf.stack([b, y1, x0], axis=1)
    indices_d = tf.stack([b, y1, x1], axis=1)#b,3,[hw9]

    pixel_values_a = tf.gather_nd(image, indices_a)#b,
    pixel_values_b = tf.gather_nd(image, indices_b)
    pixel_values_c = tf.gather_nd(image, indices_c)
    pixel_values_d = tf.gather_nd(image, indices_d)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    area_a = tf.expand_dims(((y1 - y) * (x1 - x)), 1)
    area_b = tf.expand_dims(((y1 - y) * (x - x0)), 1)
    area_c = tf.expand_dims(((y - y0) * (x1 - x)), 1)
    area_d = tf.expand_dims(((y - y0) * (x - x0)), 1)
    # total=(area_a+area_b+area_c+area_d)
    output = tf.add_n([area_a * pixel_values_a,
                        area_b * pixel_values_b,
                        area_c * pixel_values_c,
                        area_d * pixel_values_d])
    # mask=tf.cast(total>0,tf.float32)
    # total=mask*total+(1-mask)
    # output=output/total
    output=tf.reshape(output,[batch_size,height,width,9*channels])#b,h,w,h36
    # output=tf.reduce_max(output,-2)#b,h,w,channels
    return output
def homography_warping(input_image, homography):
    with tf.name_scope('warping_by_homography'):
        image_shape = tf.shape(input_image)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]

        # turn homography to affine_mat of size (B, 2, 3) and div_mat of size (B, 1, 3)
        affine_mat = tf.slice(homography, [0, 0, 0], [-1, 2, 3])
        div_mat = tf.slice(homography, [0, 2, 0], [-1, 1, 3])

        # generate pixel grids of size (B, 3, (W+1) x (H+1))
        pixel_grids = get_pixel_grids(height, width)
        pixel_grids = tf.expand_dims(pixel_grids, 0)
        pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
        pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))
        # return pixel_grids

        # affine + divide tranform, output (B, 2, (W+1) x (H+1))
        grids_affine = tf.matmul(affine_mat, pixel_grids)
        grids_div = tf.matmul(div_mat, pixel_grids)
        grids_zero_add = tf.cast(tf.equal(grids_div, 0.0), dtype='float32') * 1e-7 # handle div 0
        grids_div = grids_div + grids_zero_add
        grids_div = tf.tile(grids_div, [1, 2, 1])
        grids_inv_warped = tf.div(grids_affine, grids_div)
        x_warped, y_warped = tf.unstack(grids_inv_warped, axis=1)
        x_warped_flatten = tf.reshape(x_warped, [-1])
        y_warped_flatten = tf.reshape(y_warped, [-1])
        mask=(x_warped>=0)&(tf.cast(x_warped,dtype=tf.int32)<width)&(y_warped>=0)&(tf.cast(y_warped,dtype=tf.int32)<height)#b,h,w,1
        mask=tf.reshape(mask,[-1,height,width,1])
        mask=tf.cast(mask,tf.float32)
        # interpolation
        warped_image = interpolate(input_image, x_warped_flatten, y_warped_flatten)
        warped_image = tf.reshape(warped_image, shape=image_shape, name='warped_feature')
        warped_image=warped_image*mask
    # return input_image
    return grids_inv_warped,warped_image
def tf_transform_homography(input_image, homography):

	# tf.contrib.image.transform is for pixel coordinate but our
	# homograph parameters are for image coordinate (x_p = x_i + 0.5).
	# So need to change the corresponding homography parameters 
    homography = tf.reshape(homography, [-1, 9])
    a0 = tf.slice(homography, [0, 0], [-1, 1])
    a1 = tf.slice(homography, [0, 1], [-1, 1])
    a2 = tf.slice(homography, [0, 2], [-1, 1])
    b0 = tf.slice(homography, [0, 3], [-1, 1])
    b1 = tf.slice(homography, [0, 4], [-1, 1])
    b2 = tf.slice(homography, [0, 5], [-1, 1])
    c0 = tf.slice(homography, [0, 6], [-1, 1])
    c1 = tf.slice(homography, [0, 7], [-1, 1])
    c2 = tf.slice(homography, [0, 8], [-1, 1])
    a_0 = a0 - c0 / 2
    a_1 = a1 - c1 / 2
    a_2 = (a0 + a1) / 2 + a2 - (c0 + c1) / 4 - c2 / 2
    b_0 = b0 - c0 / 2
    b_1 = b1 - c1 / 2
    b_2 = (b0 + b1) / 2 + b2 - (c0 + c1) / 4 - c2 / 2
    c_0 = c0
    c_1 = c1
    c_2 = c2 + (c0 + c1) / 2
    homo = []
    homo.append(a_0)
    homo.append(a_1)
    homo.append(a_2)
    homo.append(b_0)
    homo.append(b_1)
    homo.append(b_2)
    homo.append(c_0)
    homo.append(c_1)
    homo.append(c_2)
    homography = tf.stack(homo, axis=1)
    homography = tf.reshape(homography, [-1, 9])

    homography_linear = tf.slice(homography, begin=[0, 0], size=[-1, 8])
    homography_linear_div = tf.tile(tf.slice(homography, begin=[0, 8], size=[-1, 1]), [1, 8])
    homography_linear = tf.div(homography_linear, homography_linear_div)

    warped_image = tf.contrib.image.transform(
        input_image, homography_linear, interpolation='BILINEAR')

    # return input_image
    return warped_image

def validflow(left_cam,right_cam,depth_map,max_flow=None):
    """

    :param input_image:
    :param left_cam:
    :param right_cam:
    :param depth_map: b,h,w,1
    :return:
    """
    with tf.name_scope('warping_by_homography'):
        image_shape = tf.shape(depth_map)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]
        pixel_grids = get_pixel_grids(height, width)
        
        pixel_grids = tf.expand_dims(pixel_grids, 0)
        pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
       
        pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))#b,3,[hxw]
        x_coords=tf.slice(pixel_grids,[0,0,0],[-1,1,-1])#b,1,[hxw]
        depth_flatten=tf.reshape(depth_map,(batch_size,1,-1))#b,1,[hxw]
        pixel_grids=pixel_grids*depth_flatten
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_left_inv = tf.squeeze(tf.matrix_inverse(K_left),1)
        # K_right=tf.squeeze(K_right,1)
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])
        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))  # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)#b,3,1
        middle_mat1 = tf.matmul(R_left_trans, K_left_inv)#b,3,3
        middle_mat2 = tf.squeeze(tf.matmul(K_right, R_right),1)  # b,3,3
        pixel_grids=tf.matmul(middle_mat1,pixel_grids)#b,3,[h,w]
        pixel_grids=tf.subtract(pixel_grids,c_relative)
        pixel_grids=tf.matmul(middle_mat2,pixel_grids)
        grids_div=tf.slice(pixel_grids,[0,2,0],[-1,1,-1])#b,1,[h,w]
        grids_zero_add = tf.cast(tf.equal(grids_div, 0.0), dtype='float32') * 1e-7  # handle div 0
        # grids_div = grids_div + grids_zero_add
        grids_divs = tf.tile(grids_div+ grids_zero_add, [1, 2, 1])
        grids_affine=tf.slice(pixel_grids,[0,0,0],[-1,2,-1])
        grids_inv_warped = tf.div(grids_affine, grids_divs)
        x_warped, y_warped = tf.unstack(grids_inv_warped, axis=1)
        # x_warped=tf.reshape(x_warped,[batch_size,height,width,1])
        x_warped=tf.reshape(x_warped,[batch_size,1,-1])#b,1,[hxw]
        mask=tf.abs(x_warped-x_coords)<max_flow
        mask=tf.cast(mask&(x_warped>0.0 )&(x_warped<tf.cast(width,tf.float32)),tf.float32)
        x_warped=tf.reshape(x_warped,[batch_size,height,width,1])
        x_coords=tf.reshape(x_coords,[batch_size,height,width,1])
        mask=tf.reshape(mask,[batch_size,height,width,1])

        disp=(x_warped-x_coords)*mask

        # disp=tf.cast(disp<max_flow,tf.float32)*disp
        disp=tf.reshape(disp,tf.shape(depth_map))
        return disp,mask

def reprojection(input_image,left_cam,right_cam,depth_map):
    """

    :param input_image:
    :param left_cam:
    :param right_cam:
    :param depth_map: b,h,w,1
    :return:
    """
    with tf.name_scope('warping_by_homography'):
        image_shape = tf.shape(input_image)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]
        pixel_grids = get_pixel_grids(height, width)
        pixel_grids = tf.expand_dims(pixel_grids, 0)
        pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
        pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))#b,3,[hxw]
        depth_flatten=tf.reshape(depth_map,(batch_size,1,-1))#b,1,[hxw]
        pixel_grids=pixel_grids*depth_flatten
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_left_inv = tf.squeeze(tf.matrix_inverse(K_left),1)
        # K_right=tf.squeeze(K_right,1)
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])
        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))  # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)#b,3,1
        middle_mat1 = tf.matmul(R_left_trans, K_left_inv)#b,3,3
        middle_mat2 = tf.squeeze(tf.matmul(K_right, R_right),1)  # b,3,3
        pixel_grids=tf.matmul(middle_mat1,pixel_grids)#b,3,[h,w]
        pixel_grids=tf.subtract(pixel_grids,c_relative)
        pixel_grids=tf.matmul(middle_mat2,pixel_grids)
        grids_div=tf.slice(pixel_grids,[0,2,0],[-1,1,-1])#b,1,[h,w]
        grids_zero_add = tf.cast(tf.equal(grids_div, 0.0), dtype='float32') * 1e-7  # handle div 0
        grids_div = grids_div + grids_zero_add
        grids_div = tf.tile(grids_div, [1, 2, 1])
        grids_affine=tf.slice(pixel_grids,[0,0,0],[-1,2,-1])
        grids_inv_warped = tf.div(grids_affine, grids_div)
        x_warped, y_warped = tf.unstack(grids_inv_warped, axis=1)
        x_warped_flatten = tf.reshape(x_warped, [-1])
        y_warped_flatten = tf.reshape(y_warped, [-1])
        mask = tf.reshape((x_warped >= 0.0) & (x_warped < tf.cast(width, tf.float32)) & (y_warped >= 0.0) & (
            y_warped < tf.cast(height, tf.float32)),
                          [batch_size, height, width, 1])
        mask=tf.cast(mask,tf.float32)
        # mask = tf.reshape(mask, [-1, height, width, 1])
        # x_warped=x_warped*mask
        # y_warped=y_warped*mask
        warped_image = interpolate(input_image, x_warped_flatten, y_warped_flatten)
        warped_image = tf.reshape(warped_image, shape=image_shape, name='warped_feature')
        # mask=mask*tf.cast(depth_map>0,tf.float32)
        warped_image = warped_image 
        return warped_image



def reprojection_grid(input_image,left_cam,right_cam,depth_map,scale=1.0):
    """

    :param input_image:
    :param left_cam:
    :param right_cam:
    :param depth_map: b,h,w,1
    :return:
    """
    with tf.name_scope('warping_by_homography'):
        image_shape = tf.shape(input_image)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]
        pixel_grids = get_pixel_grids(height, width)
        pixel_grids = tf.expand_dims(pixel_grids, 0)
        pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
        pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, height,width))#b,3,[hxw]
        pixel_grids=tf.transpose(pixel_grids,[0,2,3,1])
        expand_pixel_grids=tf.extract_image_patches(images=pixel_grids, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                                                rates=[1, 1, 1, 1], padding='SAME')#b,h,w,9,3
        expand_pixel_grids=tf.reshape(expand_pixel_grids,[batch_size,height,width,9,3])
        expand_pixel_grids=expand_pixel_grids*tf.expand_dims(depth_map,-1)#b,h,w,9,3
        pixel_grids=tf.transpose(tf.reshape(expand_pixel_grids,[batch_size,height*width*9,3]),[0,2,1])#b,3,[hw9]



        
        # depth_flatten=tf.reshape(depth_map,(batch_size,1,-1))#b,1,[hxw]
        # pixel_grids=tf.reshape(expand_pixel_grids,[batch_size,])
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_left_inv = tf.squeeze(tf.matrix_inverse(K_left),1)
        # K_right=tf.squeeze(K_right,1)
        scales=tf.reshape(tf.constant([scale,scale,1]),[1,1,3])
        K_left = K_left * scales
        K_right = K_right * scales
        
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])
        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))  # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)#b,3,1
        middle_mat1 = tf.matmul(R_left_trans, K_left_inv)#b,3,3
        middle_mat2 = tf.squeeze(tf.matmul(K_right, R_right),1)  # b,3,3
        pixel_grids=tf.matmul(middle_mat1,pixel_grids)#b,3,[h,w]
        pixel_grids=tf.subtract(pixel_grids,c_relative)
        pixel_grids=tf.matmul(middle_mat2,pixel_grids)
        grids_div=tf.slice(pixel_grids,[0,2,0],[-1,1,-1])#b,1,[h,w]
        grids_zero_add = tf.cast(tf.equal(grids_div, 0.0), dtype='float32') * 1e-7  # handle div 0
        grids_div = grids_div + grids_zero_add
        grids_div = tf.tile(grids_div, [1, 2, 1])
        grids_affine=tf.slice(pixel_grids,[0,0,0],[-1,2,-1])
        grids_inv_warped = tf.div(grids_affine, grids_div)
        x_warped, y_warped = tf.unstack(grids_inv_warped, axis=1)
        x_warped_flatten = tf.reshape(x_warped, [-1])
        y_warped_flatten = tf.reshape(y_warped, [-1])
        # mask = tf.reshape((x_warped >= 0.0) & (x_warped < tf.cast(width, tf.float32)) & (y_warped >= 0.0) & (
        #     y_warped < tf.cast(height, tf.float32)),
        #                   [batch_size, height, width, 1])
        # mask=tf.cast(mask,tf.float32)
        # mask = tf.reshape(mask, [-1, height, width, 1])
        # x_warped=x_warped*mask
        # y_warped=y_warped*mask
        warped_image = interpolate_grid(input_image, x_warped_flatten, y_warped_flatten)
        # warped_image = tf.reshape(warped_image, shape=image_shape, name='warped_feature')
        # mask=mask*tf.cast(depth_map>0,tf.float32)
        # warped_image = warped_image * mask
        return warped_image