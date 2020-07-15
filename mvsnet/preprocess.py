#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Training preprocesses.
"""

from __future__ import print_function

import json
import os
import time
import glob
import random
import math
import re
import sys
from itertools import combinations
from json import JSONDecoder

import cv2
import numpy as np
import tensorflow as tf
import scipy.io
import urllib
from tensorflow.python.lib.io import file_io
FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def center_image(img):
    """ normalize image input """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0,1), keepdims=True)
    mean = np.mean(img, axis=(0,1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)

def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal: 
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale
    return new_cam

def scale_mvs_camera(cams, scale=1):
    """ resize input in order to produce sampled depth map """
    for view in range(FLAGS.view_num):
        cams[view] = scale_camera(cams[view], scale=scale)
    return cams

def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def scale_mvs_input(images, cams, depth_image=None, scale=1):
    """ resize input to fit into the memory """
    for view in range(FLAGS.view_num):
        images[view] = scale_image(images[view], scale=scale)
        cams[view] = scale_camera(cams[view], scale=scale)
       

    if depth_image is None:
        return images, cams
    else:
        depth_image = scale_image(depth_image, scale=scale, interpolation='nearest')
        return images, cams, depth_image

def crop_mvs_input(images, cams, depth_image=None):
    """ resize images and cameras to fit the network (can be divided by base image size) """

    # crop images and cameras
    for view in range(FLAGS.view_num):
        h, w = images[view].shape[0:2]
        new_h = h
        new_w = w
        if new_h > FLAGS.max_h:
            new_h = FLAGS.max_h
        else:
            new_h = int(math.ceil(h / FLAGS.base_image_size) * FLAGS.base_image_size)
        if new_w > FLAGS.max_w:
            new_w = FLAGS.max_w
        else:
            new_w = int(math.ceil(w / FLAGS.base_image_size) * FLAGS.base_image_size)
        start_h = int(math.ceil((h - new_h) / 2))
        start_w = int(math.ceil((w - new_w) / 2))
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        images[view] = images[view][start_h:finish_h, start_w:finish_w]
        cams[view][1][0][2] = cams[view][1][0][2] - start_w
        cams[view][1][1][2] = cams[view][1][1][2] - start_h

    # crop depth image
    if not depth_image is None:
        depth_image = depth_image[start_h:finish_h, start_w:finish_w]
        return images, cams, depth_image
    else:
        return images, cams

def mask_depth_image(depth_image, min_depth, max_depth):
    """ mask out-of-range pixel to zero """
    # print ('mask min max', min_depth, max_depth)
    ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
    ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
    depth_image = np.expand_dims(depth_image, 2)
    return depth_image

def load_cam(file, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]
            
    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = 256
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam

def write_cam(file, cam):
    # f = open(file, "w")
    f = file_io.FileIO(file, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()

def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = str(file.readline()).rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    # scale = float(file.readline().rstrip())
    scale = float((file.readline()).rstrip())
    if scale < 0: # little-endian
        data_type = '<f'
    else:
        data_type = '>f' # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    # data = np.fromfile(file, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data

def write_pfm(file, image, scale=1):
    file = file_io.FileIO(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)  

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image_string = image.tostring()
    file.write(image_string)    

    file.close()
class Img():
    def __init__(self,image,cam,depth,dis):
        self.cam=cam
        self.image=image
        self.depth=depth
        self.dis=dis

class Item():
    
    def __init__(self):
        self.imgs=[]

    
def gen_dtu_resized_path(dtu_data_folder, mode='training'):
    """ generate data paths for dtu dataset """
    sample_list = []
    
    # parse camera pairs
    cluster_file_path = dtu_data_folder + '/Cameras/pair.txt'
    
    # cluster_list = open(cluster_file_path).read().split()
    cluster_list = file_io.FileIO(cluster_file_path, mode='r').read().split()

    # 3 sets
    training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                    45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                    74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                    101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                    121, 122, 123, 124, 125, 126, 127, 128]
    validation_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]

    data_set = []
    if mode == 'training':
        data_set = training_set
    elif mode == 'validation':
        data_set = validation_set

    # for each dataset
    for i in data_set:

        image_folder = os.path.join(dtu_data_folder, ('Rectified/scan%d_train' % i))
        cam_folder = os.path.join(dtu_data_folder, 'Cameras/train')
        depth_folder = os.path.join(dtu_data_folder, ('Depths/scan%d_train' % i))

        if mode == 'training':
            # for each lighting
            for j in range(0, 7):
                # for each reference image
                for p in range(0, int(cluster_list[0])):
                    item=[]
                    # ref image
                    ref_index = int(cluster_list[22 * p + 1])
                    ref_image_path = os.path.join(
                        image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
                    ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
                    depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
                    depth=cv2.imread(depth_image_path,-1)
                    ref_cam=load_cam(open(ref_cam_path))
                    depth_start=ref_cam[1][3,0]
                    depth_end=depth_start+(FLAGS.max_d-1)*ref_cam[1][3,1]
                    mask=((depth>=depth_start)&(depth<=depth_end)).astype(np.float32)
                    if np.count_nonzero(mask)*1.0/np.size(mask)<0.5:
                        continue
                    # item.depths.append(depth_image_path)
                    # item.images.append(ref_image_path)
                    # item.cams.append(ref_cam_path)
                    item.append((ref_image_path,ref_cam_path,depth_image_path,0))
                    ref_cam=load_cam(open(ref_cam_path))
                    ref_c=np.matmul(-ref_cam[0,:3,:3].transpose(),ref_cam[0,:3,3])
                    # view images
                    FIN_MAX=10000
                    check_num=FLAGS.view_num-1
                    # current_idx=int(ref_index)-check_num/2
                    # current_idx=np.clip(current_idx,0,int(cluster_list[0])-check_num-1)
                    current_idx=int(ref_index)-check_num/2
                    current_idx=np.clip(current_idx,0,int(cluster_list[0])-check_num-1)
                    for view in range(check_num):
                        if current_idx==int(ref_index):
                            current_idx+=1
                        view_index=current_idx
                        current_idx+=1
                        # view_index = int(cluster_list[22 * p + 2 * view + 3])
                        # current_idx=np.clip(current_idx,0,int(cluster_list[0])-check_view_num-1)
                        # score=float(cluster_list[22*p+2*view+4])
#                         if ref_index<check_num/2:
#                             view_index=ref_index+view+1
#                         elif ref_index+check_num/2+1>int(cluster_list[0]):
#                             view_index=ref_index-1-view
#                         else:
#                             view_index=ref_index+(2+view)/2*(1 if view%2==0 else -1)
                        # if score<2000:
                        #     continue
                        view_image_path = os.path.join(
                            image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
                        view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                        view_cam=load_cam(open(view_cam_path))
                        view_c=np.matmul(-view_cam[0,:3,:3].transpose(),view_cam[0,:3,3])
                        dis=abs(np.sum((ref_c-view_c)**2))
                        depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % view_index))
                        
                        
                        item.append((view_image_path,view_cam_path,depth_image_path,dis))
#                     item[1:]=sorted(item[1:],cmp=lambda x,y:cmp(-x.dis,-y.dis))
                    # depth path

                    # each ref image have 10 images
                    # depths=list(combinations(item.depths[1:], 2))
                    # cams=list(combinations(item.cams[1:], 2))
                    # images=list(combinations(item.images[1:], 2))
                    # if len(item)>=6:
                    sample_list.append(item)
                    # for i in range(len(cams)):
                    #     item_p=Item()
                    #     item_p.depths.append(item.depths[0])
                    #     item_p.depths.extend(depths[i])
                    #     item_p.cams.append(item.cams[0])
                    #     item_p.cams.extend(cams[i])
                    #     item_p.images.append(item.images[0])
                    #     item_p.images.extend(images[i])
                    #     sample_list.append(item_p)
            # pf=open("train_pair.txt",'w')
            # pf.write(json.dumps(sample_list))
            # pf.close()
        elif mode == 'validation':
            j = 3
            # for each reference image
            for p in range(0, int(cluster_list[0])):
                paths = []
                item = []
                # ref image
                ref_index = int(cluster_list[22 * p + 1])
                ref_image_path = os.path.join(
                    image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
                ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
                # paths.append(ref_image_path)
                # paths.append(ref_cam_path)
                ref_cam=load_cam(open(ref_cam_path))
                depth_start=ref_cam[1][3,0]
                depth_end=depth_start+(FLAGS.max_d-1)*ref_cam[1][3,1]
                depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
                depth=cv2.imread(depth_image_path,-1)
                mask=((depth>=depth_start)&(depth<=depth_end)).astype(np.float32)
                if np.count_nonzero(mask)*1.0/np.size(mask)<0.5:
                    continue
                # item.depths.append(depth_image_path)
                # item.images.append(ref_image_path)
                # item.cams.append(ref_cam_path)
                item.append((ref_image_path,ref_cam_path,depth_image_path,0))
                # view images
                check_num=FLAGS.view_num-1
                # current_idx=np.clip(current_idx,0,int(cluster_list[0])-check_num-1)
                current_idx=int(ref_index)-check_num/2
                current_idx=np.clip(current_idx,0,int(cluster_list[0])-check_num-1)
                for view in range(check_num):
                    if current_idx==int(ref_index):
                        current_idx+=1
                    view_index=current_idx
                    current_idx+=1
                    # if ref_index<check_num/2:
                    #     view_index=ref_index+view+1
                    # elif ref_index+check_num/2+1>int(cluster_list[0]):
                    #     view_index=ref_index-1-view
                    # else:
                    #     view_index=ref_index+(2+view)/2*(1 if view%2==0 else -1)
                    # view_index = int(cluster_list[22 * p + 2 * view + 3])
                    view_image_path = os.path.join(
                        image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
                    view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                    depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % view_index))
                    
                    # item.depths.append(depth_image_path)
                    # item.images.append(view_image_path)
                    # item.cams.append(view_cam_path)
                    item.append((view_image_path,view_cam_path,depth_image_path,0))
                    # paths.append(view_image_path)
                    # paths.append(view_cam_path)
                # item=sorted(item,cmp=lambda x,y:cmp(x.dis,y.dis))
                # depth path
                # depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
                # paths.append(depth_image_path)
                # sample_list.append(paths)
                sample_list.append(item)
            # pf=open("valid_pair.txt",'w')
            # pf.write(json.dumps(sample_list))
            # pf.close()
    return sample_list
def obj2json(obj):
    return {
        "image":obj.image,
        "cam":obj.cam,
        "depth":obj.depth
       
    }
def gen_dtu_mvs_path(dtu_data_folder, mode='training'):
    """ generate data paths for dtu dataset """
    sample_list = []
    
    # parse camera pairs
    cluster_file_path = dtu_data_folder + '/Cameras/pair.txt'
    cluster_list = open(cluster_file_path).read().split()

    # 3 sets
    training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                    45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                    74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                    101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                    121, 122, 123, 124, 125, 126, 127, 128]
    validation_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
    evaluation_set = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 
                      110, 114, 118]

    # for each dataset
    data_set = []
    if mode == 'training':
        data_set = training_set
    elif mode == 'validation':
        data_set = validation_set
    elif mode == 'evaluation':
        data_set = evaluation_set

    # for each dataset
    for i in data_set:

        image_folder = os.path.join(dtu_data_folder, ('Rectified/scan%d' % i))
        cam_folder = os.path.join(dtu_data_folder, 'Cameras')
        depth_folder = os.path.join(dtu_data_folder, ('Depths/scan%d' % i))

        if mode == 'training':
            # for each lighting
            for j in range(0, 7):
                # for each reference image
                for p in range(0, int(cluster_list[0])):
                    paths = []
                    # ref image
                    ref_index = int(cluster_list[22 * p + 1])
                    
                    ref_image_path = os.path.join(
                        image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
                    ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
                    paths.append(ref_image_path)
                    paths.append(ref_cam_path)
                    # view images
                    for view in range(FLAGS.view_num - 1):
                        view_index = int(cluster_list[22 * p + 2 * view + 3])
                        view_image_path = os.path.join(
                            image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
                        view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                        paths.append(view_image_path)
                        paths.append(view_cam_path)
                    # depth path
                    depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
                 
                    paths.append(depth_image_path)
                    sample_list.append(paths)
        else:
            # for each reference image
            j = 5
            for p in range(0, int(cluster_list[0])):
                paths = []
                # ref image
                ref_index = int(cluster_list[22 * p + 1])
                ref_image_path = os.path.join(
                    image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
                ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
                paths.append(ref_image_path)
                paths.append(ref_cam_path)
                # view images
                for view in range(FLAGS.view_num - 1):
                    view_index = int(cluster_list[22 * p + 2 * view + 3])
                    view_image_path = os.path.join(
                        image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
                    view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                    paths.append(view_image_path)
                    paths.append(view_cam_path)
                # depth path
                depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
                paths.append(depth_image_path)
                sample_list.append(paths)
        
    return sample_list

def gen_mvs_list(mode='training'):
    """output paths in a list: [[I1_path1,  C1_path, I2_path, C2_path, ...(, D1_path)], [...], ...]"""
    sample_list = []
    if FLAGS.train_dtu:
        dtu_sample_list = gen_dtu_mvs_path(FLAGS.dtu_data_root, mode=mode)
        sample_list = sample_list + dtu_sample_list
    return sample_list

# for testing
def gen_pipeline_mvs_list(dense_folder):
    """ mvs input path list """
    image_folder = os.path.join(dense_folder, 'images')
    cam_folder = os.path.join(dense_folder, 'cams')
    cluster_list_path = os.path.join(dense_folder, 'pair.txt')
    cluster_list = open(cluster_list_path).read().split()

    # for each dataset
    mvs_list = []
    pos = 1
    lens=int(cluster_list[0])
   
    for i in range(int(cluster_list[0])):
        paths = []
        item=[]
        # ref image
        # item=Item()
        ref_index = cluster_list[pos]
        if ref_index.isdigit():
            ref_index="%08d"%int(ref_index)
                                
        pos += 1
        ref_image_path = os.path.join(image_folder, (ref_index+".jpg"))
        ref_cam_path = os.path.join(cam_folder, (ref_index+'_cam.txt'))
        if os.path.exists(ref_cam_path)==False:
            continue
        paths.append(ref_image_path)
        paths.append(ref_cam_path)
        item.append(Img(ref_image_path,ref_cam_path,None,0))
        ref_cam=load_cam(open(ref_cam_path))
        ref_c=np.matmul(-ref_cam[0,:3,:3].transpose(),ref_cam[0,:3,3])

        # view images
        all_view_num = int(cluster_list[pos])
        pos += 1
        check_view_num = min(FLAGS.view_num - 1, all_view_num)
        ref_index=int(ref_index)
        current_idx=int(ref_index)-check_view_num/2
        current_idx=np.clip(current_idx,0,int(cluster_list[0])-check_view_num-1)
        for view in range(check_view_num):
            # if(int(ref_index)+check_view_num)<lens:
            #     view_index=int(ref_index)+view+1
            # else:
            #     view_index = int(cluster_list[pos + 2 * view])
            if current_idx==int(ref_index):
                current_idx+=1
            view_index=current_idx
            current_idx+=1

             
            # if view_index.isdigit():
            # view_index = int(cluster_list[pos + 2 * view])
            view_index="%08d"%int(view_index)
            view_image_path = os.path.join(image_folder, (view_index+'.jpg'))
            view_cam_path = os.path.join(cam_folder, (view_index+'_cam.txt'))
            if os.path.exists(view_cam_path)==False:
                continue
            view_cam=load_cam(open(view_cam_path))
            view_c=np.matmul(-view_cam[0,:3,:3].transpose(),view_cam[0,:3,3])
            dis=np.sum((ref_c-view_c)**2)
            
            item.append(Img(view_image_path,view_cam_path,None,dis))
            paths.append(view_image_path)
            paths.append(view_cam_path)
        # item[1:]=sorted(item[1:],cmp=lambda x,y:cmp(x.dis,y.dis))
        pos += 2 * all_view_num
        # depth path
        mvs_list.append(item)
    return mvs_list
def gen_eth3d_path(eth3d_data_folder, mode='training'):
    """ generate data paths for eth3d dataset """

    sample_list = []

    data_names = ['delivery_area', 'electro', 'forest', 'playground', 'terrains']


    for data_name in data_names:

        data_folder = os.path.join(eth3d_data_folder, data_name)

        image_folder = os.path.join(data_folder, 'images')
        depth_folder = os.path.join(data_folder, 'depths')
        cam_folder = os.path.join(data_folder, 'cams')

        # index to image name
        index2name = dict()
        dict_file = os.path.join(cam_folder,'index2prefix.txt')
        dict_list = file_io.FileIO(dict_file, mode='r').read().split()
        dict_size = int(dict_list[0])
        for i in range(0, dict_size):
            index = int(dict_list[2 * i + 1])
            name = str(dict_list[2 * i + 2])
            index2name[index] = name

        # image name to depth name
        name2depth = dict()
        name2depth['images_rig_cam4_undistorted'] = 'images_rig_cam4'
        name2depth['images_rig_cam5_undistorted'] = 'images_rig_cam5'
        name2depth['images_rig_cam6_undistorted'] = 'images_rig_cam6'
        name2depth['images_rig_cam7_undistorted'] = 'images_rig_cam7'

        # cluster
        cluster_file = os.path.join(cam_folder,'pair.txt')
        cluster_list = file_io.FileIO(cluster_file, mode='r').read().split()
        for p in range(0, int(cluster_list[0])):
            paths = []
            # ref image
            ref_index = int(cluster_list[22 * p + 1])
            ref_image_name = index2name[ref_index]
            ref_image_path = os.path.join(image_folder, ref_image_name)
            ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
            paths.append(ref_image_path)
            paths.append(ref_cam_path)
            # view images
            for view in range(FLAGS.view_num - 1):
                view_index = int(cluster_list[22 * p + 2 * view + 3])
               
               
                view_image_name = index2name[view_index]
                view_image_path = os.path.join(image_folder, view_image_name)
                view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                paths.append(view_image_path)
                paths.append(view_cam_path)
            # depth path
            image_prefix = os.path.split(ref_image_name)[1]
            depth_sub_folder = name2depth[os.path.split(ref_image_name)[0]]
            ref_depth_name = os.path.join(depth_sub_folder, image_prefix)
            ref_depth_name = os.path.splitext(ref_depth_name)[0] + '.pfm'
            depth_image_path = os.path.join(depth_folder, ref_depth_name)
            paths.append(depth_image_path)
            sample_list.append(paths)

    return sample_list
if __name__=="__main__":
    # pfm=load_pfm(open("/home/haibao637//data/tankandtemples/intermediate//Family/depths_mvsnet/00000000_prob.pfm"))
    # depth = load_pfm(open("/home/haibao637//data/tankandtemples/intermediate//Family/depths_mvsnet/00000000_init.pfm"))
    # img=cv2.imread("/home/haibao637//data/tankandtemples/intermediate//Family/depths_mvsnet/00000000_init.pfm")
    # print(pfm.shape,depth.shape,img.shape)
    sample_list=gen_dtu_resized_path("/home/haibao637/data/mvs_training/dtu/","validation")
    print(sample_list[876].images)
    # def from_json(obj):
    #     return Item(obj["images"],obj["cams"],obj["depths"])
    # file=open('log.txt','r')
    # obj=json.load(file)
    # sample_list=JSONDecoder(object_hook=from_json).decode(obj)
    print(len(sample_list))


def gen_demon_list(train_dir,mode='train'):
    if mode=='train':
        train_file=os.path.join(train_dir,'train.txt')
    else:
        train_file=os.path.join(train_dir,'val.txt')
    cluster_list = open(train_file).read().split()
    sample_list=[]
    for cluster in cluster_list:
        # if cluster.find("inf")!=-1:
        #     continue
        train_item=os.path.join(train_dir,cluster)
        poses=[float(x) for x in  open(os.path.join(train_item,'poses.txt')).read().split()]
        poses=np.array(poses).reshape(-1,3,4)
        intrinsic=[float(x) for x in  open(os.path.join(train_item,'cam.txt')).read().split()]
        intrinsic=np.array(intrinsic).reshape(3,3)
        
        item=[]
        for index in range(poses.shape[0]) :
            img=os.path.join(train_item,"%04d.jpg"%index)
            depth=os.path.join(train_item,"%04d.npy"%index)
            cam=np.zeros([2,4,4])
            cam[0,:3,:]=poses[index]
            cam[1,:3,:3]=intrinsic
            c=np.matmul(-cam[0,:3,:3].transpose(),cam[0,:3,3])
            if len(item)==0:
                item.append(Img(img,cam,depth,0))
            # dis=np.abs(np.math.acos(np.matmul(item[0].cam[0,2,:3],cam[0,2,:3].reshape(3,1)))-np.math.pi/3)
            item.append(Img(img,cam,depth,0))
            
        if len(item)<3:
            continue
        item[1:]=sorted(item[1:],cmp=lambda x,y:cmp(x.dis,y.dis))
        # print(item)
        # items=[[item,x[0],x[1]] for x in list(combinations(item[1:], 2))]
        # items=list(combinations(item[1:], 3))
        sample_list.append(item)
    return sample_list


        