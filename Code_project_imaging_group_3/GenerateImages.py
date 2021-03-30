from tensorflow.keras import Model
import numpy as np
from PIL import Image
import tensorflow as tf

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

def GenerateImages(model_path):
    """
    Code that saves the fake images, to save the images create two folders one titled 0 and one titled 1 to sort the images into with and without metastases
    Parameters: model_path, the path where the model is located
    """
    #load the model 
    generator = tf.keras.models.load_model(model_path)
    batch=0
    #The amount of images in a batch
    N_batch = 64
    #Amount of batches for which the model is run
    nr_batches  = 1
    for i in range(nr_batches):
        Fake_images, label = Generate_Fake_Images(generator,batch_size=N_batch)
        for j in range(N_batch):
            image = (Fake_images[j] * 255).astype(np.uint8)
            image = Image.fromarray(image,'RGB')
            #Save each image as Fake_image_(imagenumber) in either the 0 or 1 folder
            image.save('%1d/Fake_image_%07d.jpg' % (int(label[j]),N_batch*batch+j))
    batch +=1
    #print the batch number to monitor how far the process is
    print(batch)

GenerateImages(r'C:\Users\20174069\Desktop\Project imaging\Code_project_imaging_group_3\generator_model_075_final')