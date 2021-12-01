#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
import numpy as np
from time import time
from time import sleep


# In[3]:
def pool3D(arr, downscaling_ratio,
           func=np.nanmax,
           ):
    # check inputs
    #assert arr.ndim == 3
    #assert len(kernel) == 3

    # create array with lots of padding around it, from which we grab stuff (could be more efficient, yes)
    kernel=(downscaling_ratio, downscaling_ratio, 1)
    stride=(downscaling_ratio, downscaling_ratio, 1)
    
    arr_padded_shape = arr.shape + 2 * np.array(kernel)
    arr_padded = np.zeros(arr_padded_shape, dtype=arr.dtype) * np.nan
    arr_padded[
    kernel[0]:kernel[0] + arr.shape[0],
    kernel[1]:kernel[1] + arr.shape[1],
    kernel[2]:kernel[2] + arr.shape[2],
    ] = arr

    # create temporary array, which aggregates kernel elements in last axis
    size_x = 1 + (arr.shape[0]-1) // stride[0]
    size_y = 1 + (arr.shape[1]-1) // stride[1]
    size_z = 1 + (arr.shape[2]-1) // stride[2]
    size_kernel = np.prod(kernel)
    arr_tmp = np.empty((size_x, size_y, size_z, size_kernel), dtype=arr.dtype)

    # fill temporary array
    kx_center = (kernel[0] - 1) // 2
    ky_center = (kernel[1] - 1) // 2
    kz_center = (kernel[2] - 1) // 2
    idx_kernel = 0
    for kx in range(kernel[0]):
        dx = kernel[0] + kx - kx_center
        for ky in range(kernel[1]):
            dy = kernel[1] + ky - ky_center
            for kz in range(kernel[2]):
                dz = kernel[2] + kz - kz_center
                arr_tmp[:, :, :, idx_kernel] = arr_padded[
                                               dx:dx + arr.shape[0]:stride[0],
                                               dy:dy + arr.shape[1]:stride[1],
                                               dz:dz + arr.shape[2]:stride[2],
                                               ]
                idx_kernel += 1

    # perform pool function
    arr_final = func(arr_tmp, axis=-1)
    return arr_final

class DataLoader():
	def load_data_test (self,path,downscaling_ratio):
		#static check =0
		#if check==0:
			#check=1
			folder = path

			onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

			print("Working with {0} images".format(len(onlyfiles)))
			print("Image examples: ")

			#for i in range(10):
			   # print(onlyfiles[i])
			    #display(_Imgdis(filename=folder + "/" + onlyfiles[i], width=119, height=99))


			# In[ ]:


			from scipy import ndimage
			from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
			from matplotlib import pyplot

			train_files = []
			y_train = []
			i=0



			
			for _file in onlyfiles:
			    train_files.append(_file)
			    label_in_file = _file.find("_")
			    #y_train.append(int(_file[0:label_in_file]))
			    
			print("Files in train_files: %d" % len(train_files))
			train_files=sorted(train_files)
			print(train_files)
			# Original Dimensions
			image_width = 512
			image_height = 512
			ratio =downscaling_ratio
			#1.7578125*2*2
			#ratio2=4

			image_width = int(image_width / ratio)
			image_height = int(image_height / ratio)

			channels = 3
			nb_classes = 1

			dataset = np.ndarray(shape=(len(train_files), image_height, image_width,channels),
					     dtype=np.float32)

			i = 0
			size=int(512/downscaling_ratio)
			for _file in train_files:
			    
			    img = load_img(folder + "/" + _file)  # this is a PIL image
			    newsize = (size,size) 
			  
			    # Convert to Numpy Array
			    x = img_to_array(img)  
			    x=pool3D(x,downscaling_ratio)
			    #x = x.reshape((size,size,3))
			    # Normalize
			    x = (x) / 255.0
			    dataset[i] = x
			    i += 1
			    if i % 250 == 0:
			    	print("%d images to array" % i)
			print("All images to array!")

			print (dataset.shape)
			imgs_lr=dataset

###########################################################################################
			
			return 5,imgs_lr,train_files

	def load_data_init (self,path):
		#static check =0
		#if check==0:
			#check=1
			folder = path

			onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

			print("Working with {0} images".format(len(onlyfiles)))
			print("Image examples: ")

			for i in range(10):
			    print(onlyfiles[i])
			    display(_Imgdis(filename=folder + "/" + onlyfiles[i], width=900, height=900))


			# In[ ]:


			from scipy import ndimage
			from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
			from matplotlib import pyplot

			train_files = []
			y_train = []
			i=0



			for _file in onlyfiles:
			    train_files.append(_file)
			    label_in_file = _file.find("_")
			    #y_train.append(int(_file[0:label_in_file]))
			    
			print("Files in train_files: %d" % len(train_files))
			train_files=sorted(train_files)
			print("train_files")
			# Original Dimensions
			image_width = 512
			image_height = 512
		

			channels = 3
			nb_classes = 1

			dataset = np.ndarray(shape=(len(train_files), image_height, image_width,channels),
					     dtype=np.float32)

			i = 0
			size=512
			for _file in train_files:
			    
			    img = load_img(folder + "/" + _file)  # this is a PIL image
			    newsize = (size,size) 
			    
			    img = img.resize(newsize)
			    #display(img)
			    # Convert to Numpy Array
			    x = img_to_array(img)  
			    x = x.reshape((size,size,3))
			    # Normalize
			    x = (x) / 255.0
			    dataset[i] = x
			    i += 1
			    if i % 250 == 0:
			    	print("%d images to array" % i)
			print("All images to array!")

			print (dataset.shape)
			imgs_hr=dataset

			###############################################################################################
			
			return imgs_hr,5
			
	def load_data(self,imgs_hr_data,imgs_lr_data,batch_size=1):
			
			k=np.random.randint(0,5023)


			return imgs_hr_data[k:k+batch_size,:,:,:], imgs_lr_data[k:k+batch_size,:,:,:]



'''
n_samples=50-4
n_frames=4
row=128
col=128

images_train = np.zeros((n_samples, n_frames, row, col, 3), dtype=np.float)
next_image_in_train = np.zeros((n_samples, n_frames, row, col, 3), dtype=np.float)

for i in range(50-5):
    for j in range(4):
        images_train[i,j,:,:,:]=dataset[i+j,:,:,:]
        next_image_in_train[i,j,:,:,:]=dataset[i+j+1,:,:,:]
      
for i in range(4):
    display(array_to_img(images_train[0,i]))
print('..')
for i in range(4):
    display(array_to_img(next_image_in_train[0,i]))

'''
