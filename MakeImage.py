from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import tensorflow as tf

def plotImages(images, label,dim=(10,10), figsize=(10, 10), title=''):
    """
    Function that plots images from the dataset

    Parameters: images: dataset of a collection of images
                dim: tuple which shows the dimensions of the subplot
                figsize: tuple which shows the figure size
                title: string in where the title of the generated plot can be added

    Returns: a plot of the images
    """
    n = images.shape[0]//2
    fig, axes = plt.subplots(2,n,figsize = (n*4,7))
    plt.axis('off')
    zero = 0
    one = 0
    for i in range(images.shape[0]):
        if label[i] == 0:
            axes[0,zero].imshow(images[i])
            axes[0,zero].set_title('Without Metastases')
            zero = zero+1
        else:
            axes[1,one].imshow(images[i]) 
            axes[1,one].set_title('With Metastases')
            one = one+1
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()
def Generate_Fake_Images(generator,batch_size=64,latent_dim=100):
    """
    Function that generates fake images using the trained model

    Parameters:generator: the generator model
           batch_size: the amount of images that will be created
           latent_dim: integer that shows the latent dimensions
    Returns:
    generated_images: the generated fake images
    gen_labels: the labels of the generated fake images
    """
    geninput = np.random.normal(0, 1, size=[batch_size, latent_dim])
    # half of the images are made to be 0 half are 1
    genlabels = np.zeros((batch_size,))
    genlabels[:batch_size//2] = 1
    np.random.shuffle(genlabels)
    # use the generator to generate new random images from the generator input
    generated_images = generator.predict([geninput,genlabels])
    generated_images = (generated_images+1.0)/2.0
    return generated_images, genlabels 

generator = tf.keras.models.load_model('generator_model_120')
fake_images,labels = Generate_Fake_Images(generator,batch_size=16)
plotImages(fake_images,labels)
batch=0
#Code that saves the fake images
#To save the images create two folders one titled 0 and one titled 1 to sort the images into with and without metastases
N_batch = 64
# #Amount of batches of images that are created
# for i in range(1024):
#   Fake_images, label = Generate_Fake_Images(generator,batch_size=N_batch)
#   for j in range(N_batch):
#     image = (Fake_images[j] * 255).astype(np.uint8)
#     image = Image.fromarray(image,'RGB')
#     image.save('%1d/Fake_image_%07d.jpg' % (int(label[j]),N_batch*batch+j))
#   batch +=1
#   print(batch)