# Super-Resolution of Sea Surface Temperature Satellite Images using PF-GAN-SR

We formulate the super-resolution problem as one of data assimilation. Data assimilation is a technique where observational data is combined with predictions from a dynamical model to optimally estimate the true state of a system. The super-resolution problem is posed as a static data assimilation problem that requires an accurate prior probability distribution estimate for the high-resolution data, an observation model and low-resolution data and an algorithm to compute the posterior and estimate the true high-resolution data.

## Dataset 
We used the 5.5 km resolution OSTIA SST data set from 2006 to 2019 (total of 5023 SST fields) for our experiments. From this data set, high-resolution and different low-resolution data sets are created for our downscaling experiments. 

## PF-GAN-SR
Our algorithm has two different components: GAN and particle filter 
1. First a GAN is trained to generate more samples from the prior probability distribution of high resolution images.</br>
We used a Pro-GAN arcitecture for our experiments. To train the proGAN first save all your data in a single directory. Then use ProGAN/progressive_gan.py </br>
use FID_score.py to evaluate the results from GAN 
2. Then a particle filter algorithm is used to generate the posterior from the samples generated during step 1 and the input low resolution field.</br>
you can run particle_filter_ostia_gen.py with the downscaling parameter as an argument.

## Evaluation
Two error metrics are used to assess the skill of the new PF-GAN-SR, bi-cubic interpolation and SRGAN as follows
1. SSIM (structural Similarity score)
2. RMSE (Root mean square error)
Use SSIM_PSNR_MSSSIM.py to get the evaluation metrics

