#-*- coding:utf-8 -*-
import imageio
import torch.utils.data.dataset as dataset
import os
import cv2
import numpy as np


def center_image(img):
    """ normalize image input """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0,1), keepdims=True)
    mean = np.mean(img, axis=(0,1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)
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


import math
def crop(image, cams, max_h,max_w):
	""" resize images and cameras to fit the network (can be divided by base image size) """

	# crop images and cameras

	h, w = image.shape[0:2]
	new_h = h
	new_w = w
	if new_h > max_h:
		new_h = max_h
	else:
		new_h = int(math.ceil(h / 32) * 32)
	if new_w > max_w:
		new_w = max_w
	else:
		new_w = int(math.ceil(w / 32) * 32)
	start_h = int(math.ceil((h - new_h) / 2))
	start_w = int(math.ceil((w - new_w) / 2))
	finish_h = start_h + new_h
	finish_w = start_w + new_w
	image = image[start_h:finish_h, start_w:finish_w]
	cams[1][0][2] = cams[1][0][2] - start_w
	cams[1][1][2] = cams[1][1][2] - start_h

	return image, cams
def depth2norm(d_im,K):
	# zy, zx = np.gradient(d_im)
	# You may also consider using Sobel to get a joint Gaussian smoothing and differentation
	# to reduce noise
	#zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
	#zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

	inv_K=np.linalg.inv(K[:3,:3])
	height=d_im.shape[0]
	width=d_im.shape[1]
	coord_buffer = np.ones((height, width, 3), dtype=np.float32)
	coord_buffer[..., 1], coord_buffer[..., 0] = np.mgrid[0:height, 0:width]
	coord_buffer*=d_im
	#coord_buffer h,w,3
	coord_buffer=np.matmul(inv_K,coord_buffer[...,np.newaxis])[...,0]#h,w,3
	dx,dy,_=np.gradient(coord_buffer)
	# dx h,w,3  dy h,w,3
	normal=np.cross(dx,dy)
	# mask=normal[...,2]<=0
	# #
	# normal[mask,:]*=-1
	# normal=np.normalize_axis_index(normal,ndim=-1)
	# m=np.linalg.norm(normal, axis=-1)
	# mask=m>0
    #
	# normal[mask,:]=normal[mask,:]/m[...,np.newaxis][mask,:]
	# normal[mask==False,:]=0.0
	# normal[mask[...,0],:]*=0.0
	return normal

class NccDataSet(dataset.Dataset):
	def __init__(self,data_dir,view_num=3):
		self.data_dir=data_dir
		self.view_num=view_num
		self.cam_dir=os.path.join(self.data_dir,"cams")
		self.image_dir=os.path.join(self.data_dir,"images/")
		self.depth_dir=os.path.join(self.data_dir,"depths_mvsnet//")
		self.image_dir=self.depth_dir
		self.cam_dir=self.depth_dir
		# self.images=sorted([f for f in os.listdir(self.depth_dir) if  f.endswith(".jpg")])
		# self.depths = sorted([f for f in os.listdir(self.depth_dir) if f.endswith(".exr")])
		# self.cameras = sorted([f for f in os.listdir(self.depth_dir) if f.endswith(".txt")])
		# self.relations=[]
		cluster_list = open(os.path.join(self.data_dir,"pair.txt"), mode='r').read().split()
		self.mvs_list = []
		pos = 1
		for i in range(int(cluster_list[0])):
			paths = []
			# ref image
			ref_name = (cluster_list[pos])
			if ref_name.isdigit():
				ref_name = "{:08d}".format(int(ref_name))
			pos += 1
			ref_image_path = os.path.join(self.image_dir, (ref_name + ".jpg"))

			ref_cam_path = os.path.join(self.cam_dir, (ref_name + '.txt'))
			paths.append(ref_image_path)
			paths.append(ref_cam_path)
			# view images
			all_view_num = int(cluster_list[pos])
			pos += 1
			check_view_num = min(view_num - 1, all_view_num)
			if check_view_num<=0:
				continue
			for view in range(check_view_num):
				n_name = (cluster_list[pos + 2 * view])
				if n_name.isdigit():
					n_name = "{:08d}".format(int(n_name))
				view_image_path = os.path.join(self.image_dir, (n_name + ".jpg"))

				view_cam_path = os.path.join(self.cam_dir, (n_name + '.txt'))
				paths.append(view_image_path)
				paths.append(view_cam_path)
			pos += 2 * all_view_num

			paths.append(os.path.join(self.depth_dir,(ref_name+".exr")))
			# depth path

			self.mvs_list.append(paths)
		# self.mvs_list=self.mvs_list[144:]


	def __len__(self):
		return len(self.mvs_list)

	def __getitem__(self, pos):
		data=self.mvs_list[pos]
		# print(data)
		selected_view_num = int(len(data) // 2)
		images=[]
		cams=[]
		ref_cam=load_cam(open(data[1], mode='r'))
		ref_cam[1,3,:1]=0
		ref_cam[1,3,-1]=1
		inv_R=np.linalg.inv(ref_cam[0,:3,:3])
		# print(ref_cam[1].shape)
		ref_image = cv2.imread(data[0])
		ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
		# ref_image = center_image(ref_image)
		# depth_map=ref_image*.0
		ref_image=ref_image[...,np.newaxis]/255.0
		#
		depth_map = np.array(imageio.imread(data[-1]))  # 1,n,n

		# depth_map = np.sign(np.linalg.det(ref_cam[0])) * depth_map * np.linalg.norm(ref_cam[0][2, :3])
		# max_h, max_w = depth_map.shape[0] *2,depth_map.shape[1]*2
		# ref_image, ref_cam = crop(ref_image, ref_cam, max_h, max_w)
		# p0=np.linalg.inv(np.matmul(ref_cam[1,:3,:3],ref_cam[0,:3,:3]))
		# depth_map=np.random.rand(ref_image.shape[0],ref_image.shape[1])*0.1
		# RC=-np.matmul(np.linalg.inv(ref_cam[0,:3,:3]),ref_cam[0,:3,3]) # 3,1
		norm=depth2norm(depth_map[...,np.newaxis],ref_cam[1])

		# norm*=0.0
		# norm[...,2]=-1.0
		Ts=[]
		Ks=[]
		Rs=[]
		#
		for view in range(1,min(self.view_num, selected_view_num)):
			image = cv2.imread(data[2 * view])

			# image_file = file_io.FileIO(data[2 * view], mode='r')
			# image = scipy.misc.imread(image_file, mode='RGB')
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			# image=center_image(image)
			image=image[...,np.newaxis]/255.0

			# cam = load_cam(open(data[2 * view + 1]))
			cam = load_cam(open(data[2 * view + 1], mode='r'))
			# image,cam=crop(image,cam,max_h,max_w)
			# p1=np.matmul(cam[1][:3,:3],cam[0][:3,:3])
			# C=-np.matmul(np.linalg.inv(cam[0,:3,:3]),cam[0,:3,3])
			# DC=C-RC
			# NC=np.matmul(DC[...,np.newaxis],ref_cam[0,2,:3][np.newaxis,...]) # 3,3
			# kt = np.matmul(p1,np.matmul(NC,p0))#3,3
			# kt=np.matmul(cam[1,:3,:3],np.matmul(cam[0,:3,:3],DC))
			# cam=np.matmul(p1,p0)
			images.append(image)

			Rs.append((np.matmul(inv_R,cam[0,:3,:3])))
			T=cam[0,:3,3]-np.matmul(  np.matmul(cam[0,:3,:3],np.linalg.inv(ref_cam[0,:3,:3])),ref_cam[0,:3,3])
			Ts.append(T)
			Ks.append([np.linalg.inv(ref_cam[1,:3,:3]),cam[1,:3,:3]])

		Ts=np.stack(Ts,0)
		Rs=np.array(Rs)
		Ks=np.array(Ks)
		images=np.stack(images, axis=0) # n,w,h,3
		# cams=np.stack(cams,axis=0) # n,4,4
		# print("det",np.linalg.det(ref_cam[0]),np.linalg.norm(ref_cam[0][2,:3]))
		norm=norm.transpose(2,0,1)

		return {
			"ref_name":data[0],
			"ref_image":ref_image.astype(np.float32),
			"images":images.astype(np.float32),
			# "cams":cams.astype(np.float32),
			"Ts":Ts[...,np.newaxis].astype(np.float32),
			"Rs": Rs.astype(np.float32),
			"Ks": Ks.astype(np.float32),
			"depth_map":depth_map[np.newaxis,...].astype(np.float32),
			"norm_map":norm.astype(np.float32)
		}

