import matplotlib
import sys
matplotlib.use('Agg')
from load_sst_data_pf_gen import DataLoader
import numpy as np
import random
from PIL import Image
import skimage.measure
import matplotlib.pyplot as plt
import scipy
data_loader = DataLoader()
from numpy import asarray
from numpy import savetxt
import time
start=time.time()

downscaling_ratio=sys.argv[1]
downscaling_ratio=int(downscaling_ratio)

path1="GAN_10000"
#path2="/home/devyani/Desktop/Mtech Research project/PROGAN/progan_models/models/interp_animation_frames"
path_test="OSTIA_test/high_res"
imgs_hr_data,_=data_loader.load_data_init(path1)
#imgs_hr_data2,_=data_loader.load_data_init(path2)
#imgs_hr_data=np.concatenate(imgs_hr_data1,imgs_hr_data2)
imgs_hr_data_test,imgs_lr_data,train_files=data_loader.load_data_test(path_test,downscaling_ratio)
print('final size', imgs_hr_data.shape)

print(imgs_lr_data.shape)


def random_sampling(n,imgs_hr_data):
	prior=[]	
	#data=np.zero((n,size,size,3))
	for i in range(n):
		#i=random.randint(0,999)
		prior.append(imgs_hr_data[i])
	
	return prior



def pool3D(arr,
           kernel=(downscaling_ratio, downscaling_ratio, 1),
           stride=(downscaling_ratio, downscaling_ratio, 1),
           func=np.nanmax,
           ):
    # check inputs
    #assert arr.ndim == 3
    #assert len(kernel) == 3

    # create array with lots of padding around it, from which we grab stuff (could be more efficient, yes)
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

def h(x):
	x=pool3D(x)
	return x

def update(x,z,w,mean_square_error):
	for i in range(n_particles):
		yhx=(z-h(x[i])).flatten()
		p_y_given_hx=np.dot(np.dot((yhx.T),mean_square_error),yhx)
		w[i]=w[i]*p_y_given_hx	
	return w

def expected_a_priori(weight_new,res,t):
	result=np.zeros((512,512,3))
	for i in range(len(weight_new)):
		result+=weight_new[i]*data[res[i]]
	print(result.shape)

	x=np.array(result)#,dtype=np.uint8)
	x=x*255
	image_pil = Image.fromarray(np.uint8(x)).convert('RGB')
	#im1 = Image.fromarray(image_pil,mode = 'RGB')
	file_name='particle_filter_10000_onlyGAN/'+str(downscaling_ratio)+'x/EAP/'+str(train_files[k])
	image_pil.save(file_name,"PNG",dpi=(300,300))

	#im = Image.fromarray(result,'RGB')
	#im.save("your_file.jpeg")
	return result

n_particles=imgs_hr_data.shape[0]

data=random_sampling(n_particles,imgs_hr_data)
data=np.array(data)
print(data.shape)

size=int(512/downscaling_ratio)

r=[0.05 for i in range(size*size*3)]
r=np.diag(r)

sum_arr=[0 for i in range(size*size*3)]
for i in range(n_particles):
	sum_arr=sum_arr+ h(data[i]).flatten()
sum_arr=sum_arr/n_particles

mean_square_error=[0 for i in range(size*size*3)]
for i in range(n_particles):
	mean_square_error=mean_square_error+np.multiply((h(data[i]).flatten()-sum_arr),(h(data[i]).flatten()-sum_arr))

mean_square_error=np.diag((0.05*mean_square_error)/n_particles)
print(mean_square_error.shape)

weight_array=[]
position_array=[]

for k in range(imgs_lr_data.shape[0]):
	print(k)
	weight=np.ones((n_particles))
	weight=weight/n_particles
	weight=update (data,imgs_lr_data[k],weight,mean_square_error)
	weight=weight/np.sum(weight)
	print(weight)
	
	#max_w = max(weight)
	maxpos=np.argmin(weight)
	t=n_particles
	N=t
	res = sorted(range(len(weight)), key = lambda sub: weight[sub])[:N]
	#print(res.shape)
	position_array.append(res)
	weight_new=[]
	for i in range(len(res)):
		x=res[i]
		value=weight[x]
		weight_new.append(1/value)
		
	
	#print(weight_new.shape)		
	#for i in range(n_particles):
	#	if(weight[i]<0.0001):
	#		weight[i]=0
	weight_new=weight_new/sum(weight_new)
	weight_array.append(weight_new)
	print(weight_new)
	expected_a_priori(weight_new,res,t)
	print('maxpos',maxpos)
	print(data[maxpos].shape)
	#res=res.reshape((256,256,3))
	x=np.array(data[maxpos])#,dtype=np.uint8)
	x=x*255
	image_pil = Image.fromarray(np.uint8(x)).convert('RGB')
	#im1 = Image.fromarray(image_pil,mode = 'RGB')
	
	file_name='particle_filter_10000_onlyGAN/'+str(downscaling_ratio)+'x/MAP/'+str(train_files[k])
	image_pil.save(file_name,"PNG",dpi=(300,300))
        
weight_array=asarray(weight_array)
savetxt('weights_'+str(downscaling_ratio)+'.csv', weight_array, delimiter=',')
position_array=asarray(position_array)
savetxt('positions_'+str(downscaling_ratio)+'.csv', position_array, delimiter=',')
end=time.time()
print(end-start)

