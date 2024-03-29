
import torch as th
import torchvision as tv
import pro_gan_pytorch_new.gan as pg
import sys 

# select the device to be used for training
#device = th.device("cuda" if th.cuda.is_available() else "cpu")
#device = th.device("cpu")
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)
             
data_path = "/home/devyani/progan_pytorch/OSTIA_512"
def setup_data():
    """
    setup the CIFAR-10 dataset for training the CNN
    :param batch_size: batch_size for sgd
    :param num_workers: num_readers for data reading
    :param download: Boolean for whether to download the data
    :return: classes, trainloader, testloader => training and testing data loaders
    """
    # data setup:
    classes = ('class1','class2')
    #classes = th.tensor((1,2)).to(device)

    # data setup:
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
             #  'dog', 'frog', 'horse', 'ship', 'truck')
    transforms = tv.transforms.ToTensor()

    trainset = tv.datasets.ImageFolder(root=data_path,
                                   transform=transforms,
                                   )
    #print(trainset)
    testset = tv.datasets.ImageFolder(root=data_path,
                                  transform=transforms
                                  )
    return classes, trainset, testset


if __name__ == '__main__':

    # some parameters:
    depth = 8
    # hyper-parameters per depth (resolution)
    num_epochs = [100, 100, 200 ,200 ,400 ,400, 500, 500]
    #num_epochs=[1,1,1,1,1,1,1,1]
    fade_ins = [50, 50, 50, 50 ,50, 50 ,50,50]
    batch_sizes = [128, 128, 128,128 ,64 ,64,32,32]
    latent_size = 256
    
    # get the data. Ignore the test data and their classes
    _, dataset, _ = setup_data()
    
    print(dataset)

    # ======================================================================
    # This line creates the PRO-GAN
    # ======================================================================
    pro_gan = pg.ProGAN(  
                                   latent_size=latent_size, device=device)
    # ======================================================================

    # ======================================================================
    # This line trains the PRO-GAN
    # ======================================================================
    pro_gan.train(
        dataset=dataset,
        epochs=num_epochs,
        fade_in_percentage=fade_ins,
        batch_sizes=batch_sizes,
        feedback_factor=10,
        )
   
    # ======================================================================  

