from __future__ import print_function
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import numpy as np
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from matplotlib import pyplot
import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
import numpy as np
from time import time
import tensorflow as tf
import math
from matplotlib import pyplot as plt
#sess = tf.InteractiveSession()

def load_data (is_real=False):

			if(is_real==True):
				folder = "OSTIA_test/high_res"
			else:
				folder = "particle_filter_10000_onlyGAN/8x/MAP"

			onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

			train_files = []
			y_train = []
			for _file in onlyfiles:
			    train_files.append(_file)
			    label_in_file = _file.find("_")
			    
			print("Files in train_files: %d" % len(train_files))
			train_files=sorted(train_files)
			print("train_files")
			# Original Dimensions
			image_width = 512
			image_height = 512
			channels = 3

			dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
					     dtype=np.float32)

			i = 0
			size=512
			for _file in train_files:
			    
			    img = load_img(folder + "/" + _file)  # this is a PIL image
			    newsize = (size,size) 
			    # Convert to Numpy Array
			    x = img_to_array(img)  
			    x = x.reshape((size,size,3))
			    dataset[i] = x
			    i += 1
			    if i % 250 == 0:
			    	print("%d images to array" % i)
			print("All images to array!")
			print (dataset.shape)
			return dataset

def calculateDistance(i1, i2):
    return math.sqrt(np.sum((i1-i2)**2))
    


X=load_data(is_real=True)
print(X)
Y=load_data()
print(X.shape)
print(Y.shape)
mse=0
arr_mse=[]
arr_ssim=[]
for i in range(186):
	mse+=calculateDistance(X[i],Y[i])
	arr_mse.append(calculateDistance(X[i],Y[i])/(512*512))
mse/=186
print(mse)
X_tensor = tf.convert_to_tensor(X)
Y_tensor = tf.convert_to_tensor(Y)

ssim = tf.image.ssim(X_tensor, Y_tensor , max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
arr_ssim=ssim.numpy()

mean=tf.reduce_mean(ssim)
mean_val = mean.numpy()
print(mean_val)

psnr = tf.image.psnr(X_tensor, Y_tensor, max_val=255)
mean=tf.reduce_mean(psnr)
mean_val = mean.numpy()
print(mean_val)


msssim = tf.image.ssim_multiscale(X_tensor, Y_tensor, max_val=255, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03)
mean=tf.reduce_mean(msssim)
mean_val = mean.numpy()
print(mean_val)

arr=[i+1 for i in range(186)]
'''plt.plot(arr,arr_ssim)
plt.xlabel('image')
plt.ylabel('SSIM')
plt.savefig('SSIM_distribution')
plt.show()'''
plt.plot(arr,arr_mse)
plt.xlabel('image')
plt.ylabel('Euclidean distance')
plt.savefig('Euclidean distance distribution')
#proto_tensor = tf.make_tensor_proto(ssim1)  # convert `tensor a` to a proto tensor
#ssim=tf.make_ndarray(proto_tensor) 
#print(ssim.mean())
'''
# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)  
X=np.random(2*3*5*5)
Y=np.random(2*3*5*5)
X=np.reshape((2,3,5,5))
Y=np.reshape((2,3,5,5))
# calculate ssim & ms-ssim for each image
ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)

# set 'size_average=True' to get a scalar value as loss.
ssim_loss = 1 - ssim( X, Y, data_range=255, size_average=True) # return a scalar
ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )

# reuse the gaussian kernel with SSIM & MS_SSIM. 
ssim_module = SSIM(data_range=255, size_average=True, channel=3)
ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

ssim_loss = 1 - ssim_module(X, Y)
ms_ssim_loss = 1 - ms_ssim_module(X, Y)


'''
'''
from SSIM_PIL import compare_ssim
from PIL import Image
import tensorflow as tf
import numpy as np
from math import log10, sqrt 
import cv2
from matplotlib import pyplot as plt


def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr

psnr_array=0
ssim_array=0
index=0
for i in range(0,400):
		file1='/home/devyani/Desktop/Mtech Research project/real_128/'+str(i)+'.png'
		file2='/home/devyani/Desktop/Mtech Research project/GAN_imgs/18GAN_imgs061.png'
		
		image1 = Image.open(file1)
		image2 = Image.open(file2)
		original = cv2.imread(file1) 
		compressed = cv2.imread(file2, 1) 
		psnr_score= PSNR(original, compressed) 
		ssim_value = compare_ssim(image1, image2)
		if(ssim_value>ssim_array):
			ssim_array=ssim_value
			index=i
		if(psnr_score>psnr_array):
			psnr_array=psnr_score



print(psnr_array)
print(ssim_array)
print(index)

'''		

