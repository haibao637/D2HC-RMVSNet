#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Convert MVSNet output to Gipuma format for post-processing.
"""

from __future__ import print_function

import argparse
import os
import time
import glob
import random
import math
import re
import sys
import shutil
from struct import *

import cv2
import numpy as np

# import p as plt
from utils import *
import imageio
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import visdom
vis=visdom.Visdom()
def read_gipuma_dmb(path):
    '''read Gipuma .dmb format image'''

    with open(path, "rb") as fid:
        
        image_type = unpack('<i', fid.read(4))[0]
        height = unpack('<i', fid.read(4))[0]
        width = unpack('<i', fid.read(4))[0]
        channel = unpack('<i', fid.read(4))[0]
        
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channel), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def write_gipuma_dmb(path, image):
    '''write Gipuma .dmb format image'''
    
    image_shape = np.shape(image)
    width = image_shape[1]
    height = image_shape[0]
    if len(image_shape) == 3:
        channels = image_shape[2]
    else:
        channels = 1

    if len(image_shape) == 3:
        image = np.transpose(image, (2, 0, 1)).squeeze()

    with open(path, "wb") as fid:
        # fid.write(pack(1))
        fid.write(pack('<i', 1))
        fid.write(pack('<i', height))
        fid.write(pack('<i', width))
        fid.write(pack('<i', channels))
        image.tofile(fid)
    return 

def mvsnet_to_gipuma_dmb(in_path, out_path,scale=1.0):
    '''convert mvsnet .pfm output to Gipuma .dmb format'''
    
    image = load_pfm(open(in_path))
    # if scale!=1:
    image=cv2.resize(image,(int(image.shape[1]*scale),int(image.shape[0]*scale)))
    write_gipuma_dmb(out_path, image)

    return 

def mvsnet_to_gipuma_cam(in_path, out_path,scale=1.0):
    '''convert mvsnet camera to gipuma camera format'''

    cam = load_cam(open(in_path))

    extrinsic = cam[0:4][0:4][0]
    intrinsic = cam[0:4][0:4][1]
    intrinsic[3][0] = 0
    intrinsic[3][1] = 0
    intrinsic[3][2] = 0
    intrinsic[3][3] = 0
    intrinsic[:2,:]*=scale
    projection_matrix = np.matmul(intrinsic, extrinsic)
    projection_matrix = projection_matrix[0:3][:]
    
    f = open(out_path, "w")
    for i in range(0, 3):
        for j in range(0, 4):
            f.write(str(projection_matrix[i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()

    return


def depth_to_normal(in_depth_path, out_normal_path):
    depth_image = load_pfm(open(in_depth_path))
    image_shape = np.shape(depth_image)

    normal_image = np.ones_like(depth_image)
    normal_image = np.reshape(normal_image, (image_shape[0], image_shape[1], 1))
    normal_image = np.tile(normal_image, [1, 1, 3])
    normal_image = normal_image / 1.732050808

    mask_image = np.squeeze(np.where(depth_image > 0, 1, 0))
    mask_image = np.reshape(mask_image, (image_shape[0], image_shape[1], 1))
    mask_image = np.tile(mask_image, [1, 1, 3])
    mask_image = np.float32(mask_image)

    normal_image = np.multiply(normal_image, mask_image)
    normal = np.float32(normal_image)

    # zy, zx = np.gradient(depth_image)
    # # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # # to reduce noise
    # # zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
    # # zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)
    #
    # normal = np.dstack((zx, zy, -np.ones_like(depth_image)))
    # n = np.linalg.norm(normal, axis=2)
    # normal[:, :, 0] /= n
    # normal[:, :, 1] /= n
    # normal[:, :, 2] /= n
    #
    #
    # # cv2.imwrite("normal.png", normal[:, :, ::-1])
    #
    imageio.imwrite(out_normal_path,normal[:,:,::-1])
    # normal += 1
    # normal /= 2
    # normal *= 255
    # vis.image(normal.transpose(2,0,1))
    return

def fake_colmap_normal(in_depth_path, out_normal_path):
    
    depth_image = read_gipuma_dmb(in_depth_path)

    # depth_image = cv2.resize(depth_image, (int(depth_image.shape[1] * scale), int(depth_image.shape[0] * scale)))
    image_shape = np.shape(depth_image)

    normal_image = np.ones_like(depth_image)
    normal_image = np.reshape(normal_image, (image_shape[0], image_shape[1], 1))
    normal_image = np.tile(normal_image, [1, 1, 3])
    normal_image = normal_image / 1.732050808

    mask_image = np.squeeze(np.where(depth_image > 0, 1, 0))
    mask_image = np.reshape(mask_image, (image_shape[0], image_shape[1], 1))
    mask_image = np.tile(mask_image, [1, 1, 3])
    mask_image = np.float32(mask_image)

    normal_image = np.multiply(normal_image, mask_image)
    normal = np.float32(normal_image)

    # zy, zx = np.gradient(depth_image)
    # # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # # to reduce noise
    # # zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
    # # zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)
    #
    # normal = np.dstack((zx, zy, -np.ones_like(depth_image)))
    # n = np.linalg.norm(normal, axis=2)
    # normal[:, :, 0] /= n
    # normal[:, :, 1] /= n
    # normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    # normal += 1
    # normal /= 2
    # normal *= 255
    #
    # cv2.imwrite("normal.png", normal[:, :, ::-1])
    write_gipuma_dmb(out_normal_path, normal)
    # imageio.imwrite()
    # normal += 1
    # normal /= 2
    # normal *= 255
    # vis.image(normal.transpose(2,0,1))
    return 

def mvsnet_to_gipuma(dense_folder, gipuma_point_folder,scale=1.0):
    
    image_folder = os.path.join(dense_folder, 'images')
    cam_folder = os.path.join(dense_folder, 'cams')
    depth_folder = os.path.join(dense_folder, 'depths_mvsnet_1')

    gipuma_cam_folder = os.path.join(gipuma_point_folder, 'cams')
    gipuma_image_folder = os.path.join(gipuma_point_folder, 'images')
    if not os.path.isdir(gipuma_point_folder):
        os.mkdir(gipuma_point_folder)
    if not os.path.isdir(gipuma_cam_folder):
        os.mkdir(gipuma_cam_folder)
    if not os.path.isdir(gipuma_image_folder):
        os.mkdir(gipuma_image_folder)

    # convert cameras 
    image_names = os.listdir(image_folder)
    for image_name in image_names:
        image_prefix = os.path.splitext(image_name)[0]
        in_cam_file = os.path.join(depth_folder, image_prefix+'.txt')
        out_cam_file = os.path.join(gipuma_cam_folder, image_name+'.P')
        mvsnet_to_gipuma_cam(in_cam_file, out_cam_file,scale)

    # copy images to gipuma image folder    
    image_names = os.listdir(image_folder)
    for image_name in image_names:
        in_image_file = os.path.join(depth_folder, image_name)
        out_image_file = os.path.join(gipuma_image_folder, image_name)
        if scale==1.0:
            shutil.copy(in_image_file, out_image_file)
        else:
            img=cv2.imread(in_image_file)
            img=cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)))
            cv2.imwrite(out_image_file,img)

    # convert depth maps and fake normal maps
    gipuma_prefix = '2333__'
    for image_name in image_names:

        image_prefix = os.path.splitext(image_name)[0]
        sub_depth_folder = os.path.join(gipuma_point_folder, gipuma_prefix+image_prefix)
        if not os.path.isdir(sub_depth_folder):
            os.mkdir(sub_depth_folder)
        in_depth_pfm = os.path.join(depth_folder, image_prefix+'_prob_filtered.pfm')
        out_depth_dmb = os.path.join(sub_depth_folder, 'disp.dmb')
        fake_normal_dmb = os.path.join(sub_depth_folder, 'normals.dmb')
        mvsnet_to_gipuma_dmb(in_depth_pfm, out_depth_dmb,scale)
        # out_depth_exr=os.path.join(depth_folder, image_prefix+'_normal.exr')
        # depth_to_normal(in_depth_pfm,out_depth_exr)
        fake_colmap_normal(out_depth_dmb, fake_normal_dmb)


from scipy import stats
import pydensecrf.densecrf as dcrf
def probability_filter(dense_folder, prob_threshold):
    image_folder = os.path.join(dense_folder, 'images')
    depth_folder = os.path.join(dense_folder, 'depths_mvsnet_1')
    
    # convert cameras 
    image_names = os.listdir(image_folder)
    for image_name in image_names:
        image_prefix = os.path.splitext(image_name)[0]

        init_depth_map_path = os.path.join(depth_folder, image_prefix+'_init.pfm')
        prob_map_path = os.path.join(depth_folder, image_prefix+'_prob.pfm')
        out_depth_map_path = os.path.join(depth_folder, image_prefix+'_prob_filtered.pfm')
#         img_path=os.path.join(depth_folder,image_prefix+".jpg")
#         img=cv2.imread(img_path)
        depth_map = load_pfm(open(init_depth_map_path))
        prob_map = load_pfm(open(prob_map_path))
        
#         sigma_xy = 80.0
#         sigma_rgb = 13.0
#         sigma_d = 2.0
#         iteration_num = 2
#         compat = np.zeros((2, 2), dtype = np.float32)
#         h,w=prob_map.shape
#         prob_maps=np.stack([prob_map,1-prob_map],0)
#         unary_energy = np.log(prob_maps)
#         crf = dcrf.DenseCRF2D(w, h, 2)
#         crf.setUnaryEnergy(-unary_energy.reshape(2, h * w))
#         # img=np.transpose(img,[1,0,2])
#         crf.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

#         crf.addPairwiseBilateral(sxy=(sigma_xy, sigma_xy), srgb=(sigma_rgb, sigma_rgb, sigma_rgb), rgbim=img, compat=10, kernel=dcrf.FULL_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
#         new_raw = crf.inference(iteration_num)
#         prob_map = np.array(new_raw).reshape(2, h,w)[0]
        
        # sorted_prob=sorted(prob_map.flatten())
        # sorted_prob_threshold=sorted_prob[int(prob_map.size*prob_threshold)]
        # sorted_prob_threshold=np.clip(sorted_prob_threshold,0.1,0.4)
        # if sorted_prob_threshold<0.1:
        #     sorted_prob_threshold=0.1
        # print(sorted_prob_threshold)
        # prob_threshold=prob_map[prob_map>0].mean()
        # depth_map[depth_map==stats.mode(depth_map)[0][0][0]]=0.0
        depth_map[prob_map <prob_threshold] = 0
        write_pfm(out_depth_map_path, depth_map)
        out_depth_map_exr_path=os.path.join(depth_folder, image_prefix+'.exr')
        imageio.imwrite(out_depth_map_exr_path,depth_map)


def depth_map_fusion(point_folder, fusibile_exe_path, disp_thresh, num_consistent):

    cam_folder = os.path.join(point_folder, 'cams')
    image_folder = os.path.join(point_folder, 'images')
    depth_min = 0.001
    depth_max = 100000
    normal_thresh = 360

    cmd = fusibile_exe_path
    cmd = cmd + ' -input_folder ' + point_folder + '/'
    cmd = cmd + ' -p_folder ' + cam_folder + '/'
    cmd = cmd + ' -images_folder ' + image_folder + '/'
    cmd = cmd + ' --depth_min=' + str(depth_min)
    cmd = cmd + ' --depth_max=' + str(depth_max)
    cmd = cmd + ' --normal_thresh=' + str(normal_thresh)
    cmd = cmd + ' --disp_thresh=' + str(disp_thresh)
    cmd = cmd + ' --num_consistent=' + str(num_consistent)
    print (cmd)
    os.system(cmd)

    return 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dense_folder', type=str, default = '/home/haibao637/data/mvsnet_input/horse/')
    parser.add_argument('--fusibile_exe_path', type=str, default = '/home/haibao637/data/fusibile/build/fusibile')
    parser.add_argument('--prob_threshold', type=float, default = '0.1')
    parser.add_argument('--disp_threshold', type=float, default = '2.5')
    parser.add_argument('--num_consistent', type=float, default = '3.0')
    parser.add_argument('--scale', type=float, default='1')
    args = parser.parse_args()

    dense_folder = args.dense_folder
    fusibile_exe_path = args.fusibile_exe_path
    prob_threshold = args.prob_threshold
    disp_threshold = args.disp_threshold
    num_consistent = args.num_consistent

    point_folder = os.path.join(dense_folder, 'points_mvsnet')
    if not os.path.isdir(point_folder):
        os.mkdir(point_folder)

    # probability filter
    print ('filter depth map with probability map')
    probability_filter(dense_folder, prob_threshold)

    # convert to gipuma format
    print ('Convert mvsnet output to gipuma input')
    mvsnet_to_gipuma(dense_folder, point_folder,args.scale)

    # depth map fusion with gipuma 
    print ('Run depth map fusion & filter')
    depth_map_fusion(point_folder, fusibile_exe_path, disp_threshold, num_consistent)
