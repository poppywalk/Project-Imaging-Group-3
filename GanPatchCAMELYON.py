# Goodfellow,  I.,  Bengio,  Y.,  and  Courville,  A.  (2016).Deep  Learning.MIT  Press.http://www.deeplearningbook.org
''' Module which makes a simple Gan, to generate new images from the dataset given
'''

import keras
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from keras.layers.convolutional import Conv2D, UpSampling2D
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, Dropout, Reshape, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Limit memory of the GPU to 2 GB
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

# define some variables
latent_dim = 100
IMAGE_SIZE = 96 
epochs =  10
batch_size = 64
batches = 0

# make two empty lists to add losses to 
discriminator_losses = []
generator_losses = []

def plotImages(images, dim=(10, 10), figsize=(10, 10), title=''):
  """
  Function that plots images from the dataset

  Parameters: images: dataset of a collection of images
              dim: tuple which shows the dimensions of the subplot
              figsize: tuple which shows the figure size
              title: string in where the title of the generated plot can be added

  Returns: a plot of the images
  """
  plt.figure(figsize=figsize)
  for i in range(images.shape[0]):
    plt.subplot(dim[0], dim[1], i+1)
    plt.imshow(images[i], interpolation='nearest')
    plt.axis('off')
  plt.tight_layout()
  plt.suptitle(title)
  plt.show()

def get_pcam_generators(base_dir, train_batch_size=32):
  """
  Function that defines the training data

  Parameters: base_dir: string in which the path where the train folder is located can be assigned
              train_batch_size: integer which shows the batch size of training data

  Returns: train_gen: directoryiterater of the training data
  """
  # dataset parameters
  train_path = os.path.join(base_dir, 'train+val', 'train')

  # define the rescaling factor for the ImageDataGenerator class 
  RESCALING_FACTOR = 1./255

  # instantiate data generators
  datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

  # collect the training data
  train_gen = datagen.flow_from_directory(train_path,
                                          target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                          batch_size=train_batch_size,
                                          class_mode='binary')

  return train_gen

def Discriminator(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64,third_filters=32,fourth_filters=64,negativeslopecoefficient=0.2):
  """
  The discriminator is a Neural network that determines if the generated images are real or fake thereby updating the generator.

  Inputs: kernel_size: tuple which shows the size of the convolutional kernel, default is (3,3)
          pool_size: integer which shows the size of the max pooling kernel, default is (4,4)
          first/second/third/fourth_filters: integers which show the amount of neurons, default is 32,64,32,64 respectively
          negativeslopecoefficient: float which shows the value of the slope coefficient for the Leaky ReLU layer, default is 0.2

  Returns: keras sequential model of the discriminator model
  """
  # begin the build of the model 
  discriminator_model = Sequential() 
  # add the first layer, a convolutional layer
  # input for this layer looks like this: (n,96,96,3)
  discriminator_model.add(Conv2D(first_filters, kernel_size, padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3))) 
  # add a leaky ReLU layer which allows a small gradient when the unit is not active
  # input for this layer looks like this (output of the first convolutional layer): (n,96,96,32)
  discriminator_model.add(LeakyReLU(negativeslopecoefficient))
  # add a maxpooling layer
  # input for this layer looks like this (output of the first Leaky ReLU layer):(n,96,96,32)
  discriminator_model.add(MaxPool2D(pool_size = pool_size))
  # add a second convolutional layer to the model
  # input for this layer looks like this (output of the first maxpooling layer): (n,24,24,32)
  discriminator_model.add(Conv2D(second_filters, kernel_size, padding = 'same'))
  # add another leaky ReLU, which allows a small gradient when the unit is not active
  # input for this layer looks like this (output of the second convolutional layer): (n,24,24,64)
  discriminator_model.add(LeakyReLU(negativeslopecoefficient))
  # add another maxpooling layer
  # input for this layer looks like this (output of the second Leaky ReLU): (n,24,24,64)
  discriminator_model.add(MaxPool2D(pool_size = pool_size))
  # flatten the output of the second maxpooling layer (n,6,6,64)
  discriminator_model.add(Flatten())
  # add a dense layer to downscale
  # input for this layer looks like this (output of flatten):(n,2304)
  discriminator_model.add(Dense(64))
  # add another Leaky ReLU, which allows a small gradient when the unit is not active
  # input for this layer looks like this (output of the first dense layer): (n,64)
  discriminator_model.add(LeakyReLU(negativeslopecoefficient))
  # add nother dense layer to downscale even more, use sigmoid which returns a value close to zero for small values,
  # and for large values the result of the function gets close to 1.
  # input for this layer looks like this (output of the third Leaky ReLU): (n,64)
  discriminator_model.add(Dense(1, activation = 'sigmoid'))
  # output of the last dense layer: (n,1)
  # model is built, now compile with an optimizer.
  # using the stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments
  discriminator_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
  return discriminator_model

def Generator(units=128*12*12, negativeslopecoefficient=0.2,size1=(2,2),size2=(4,4),kernel_size=(3, 3),genfirstfilters=64,gensecondfilters=3):
  """
  The Generator is a neural network that generates fake images

  Inputs: units: integer which shows the amount of units for the first dense layer, default is 128*12*12
          negativeslopecoefficient: float which shows the value of the slope coefficient for the Leaky ReLU layer, default is 0.2
          size1: tuple which shows the upsampling factors for rows and columns for the first UpSampling2D layer, default is (2,2)
          size2: tuple which shows the upsampling factors for rows and columns for the second UpSampling2D layer, default is (4,4)
          kernel_size: tuple which shows the size of the convolutional kernel, default is (3,3)
          genfirstfilters, gensecondfilters: integers which show the amount of neurons, default is 64,3 respectively 

  Returns: keras sequential model of the generator model
  """
  # begin the build of the model 
  generator_model = keras.models.Sequential()
  # add a dense layer 
  # input for this layer looks like this:(n,latent_dim)
  generator_model.add(Dense(units, input_dim=latent_dim, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02)))
  # add a Leaky ReLU, which allows a small gradient when the unit is not active
  # the input for this layer is (the output of the first dense layer): (n,18432)
  generator_model.add(LeakyReLU(negativeslopecoefficient))
  # add a reshape layer to shape the data
  # the input for this layer is (the output of the first LeakyReLU): (n,18342)
  generator_model.add(Reshape((12,12,128)))
  # add an UpSampling layer to repeat the rows and columns of the data by size[0] and size[1] respectively
  # the input to this layer is (the output of the Reshape layer): (n,12,12,128)
  generator_model.add(UpSampling2D(size1))
  # Conv2DTranspose can be used as well but is slower
  # add a Convolutional layer
  # the input for this layer is (the output of the UpSampling layer): (n,24,24,128)
  generator_model.add(Conv2D(genfirstfilters, kernel_size, padding='same'))
  # add another LeakyReLU layer, which allows a small gradient when the unit is not active
  # the input of this layer is (the output of the first Convolutional layer): (n,24,24,64)
  generator_model.add(LeakyReLU(negativeslopecoefficient))
  # add another UpSampling layer with a bigger size,it repeats the rows and columns of the data by size[0] and size[1] respectively 
  # the input for this layer is (the output of the second LeakyReLU): (n,24,24,64)
  generator_model.add(UpSampling2D(size2))
  # Conv2DTranspose can be used as well but is slower
  # add another Convolutional layer
  # the input for this layer is (the output of the last UpSampling layer): (n,96,96,64)
  generator_model.add(Conv2D(gensecondfilters, kernel_size, padding='same', activation='tanh'))
  # output of the last Convolutional layer: (n,96,96,3)
  # model is built, now compile with an optimizer.
  # using the stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments
  generator_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
  
  return generator_model


def Get_Gan(discriminator=Discriminator(), generator=Generator(),latent_dim=latent_dim):
  """
  Function that gets the gan

  Parameters: latent_dim: integer which shows the latent dimensions
              discriminator: keras sequential model 
              generator: keras sequential model

  Returns: Gan: model
  """
  discriminator.trainable = False
  Input = keras.layers.Input(shape=(latent_dim,))
  OutputGenerator = generator(Input)
  OutputDiscriminator = discriminator(OutputGenerator)
  gan = keras.models.Model(inputs=Input, outputs=OutputDiscriminator)
  gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
  return gan

X_train = get_pcam_generators('C:\\Users\\Kirst\\Desktop\\TUe\\8P361-Project Imaging\\Project-Imaging-Group-3',train_batch_size=batch_size)

def Train_Gan(epochs=epochs,X_train=X_train,batch_size=batch_size,latent_dim=latent_dim, discriminator=discriminator, generator=generator):
  """
  Function that trains the gan

  Parameters: epochs: integer which shows the amount of epochs the model will run
              X_train: DirectoryIterator which has the training data
              batch_size: integer which whows the size of the batch of images used per iteratoin over the data
              latent_dim: integer which shows the latent dimensions
              discriminator: keras sequential model 
              generator: keras sequential model

  Returns
  """ 
  gan = Get_Gan(discriminator, generator,latent_dim)  

  for epoch in range(epochs):

    print(f'Epoch {epoch}')
    start = time.time()

    for x_train,y_train in X_train:
      # generate random input for the generator from normal distribition (z in Goodfellow et al. 2016)
      geninput = np.random.normal(0, 1, size=[batch_size, latent_dim])
    
      # use the generator to generate new random images from the generator input
      generated_images = generator.predict(geninput)

      # concatenate the real images with the generated images, dataset with the real images followed by the fake images 
      X = np.concatenate([x_train, generated_images])

      # label the dataset, so 2 times labels of size batch_size for the generated images and the real ones
      labels_discriminator = np.zeros(2*batch_size)

      # label the real images with 1 and leave the fake to 0
      labels_discriminator[:batch_size] = 1
      
      # set the discriminator weights to trainable, so it can be trained
      discriminator.trainable = True

      # train the discriminator, and output the value of the loss function
      discriminator_loss = discriminator.train_on_batch(X, labels_discriminator)

      #  generate new input for the generator from normal distribition (z in Goodfellow et al. 2016)
      geninput = np.random.normal(0, 1, size=[batch_size, latent_dim])

      # generate ones for labels for generator for the loss function
      labels_generator = np.ones(batch_size)

      # fix the discriminator weights before training the generator, otherwise the discriminator will keep on trying to train
      discriminator.trainable = False

      # train the generator, and output the value of the loss function
      generator_loss = gan.train_on_batch(geninput, labels_generator)
      
      # add one to the batches, so it moves to the next
      batches += 1

      # print the batch number that is completed, to see if the training works
      print(batches)

      # make sure the training stops if all batches are done (imagedatagenerator loops infinitely)
      if batches >= 144000 // batch_size:
        break
    # print the epoch and how much time was needed to do the epoch
    print ('Time for epoch {} is {} sec'.format(e, time.time()-start))

    # save the checkpoints every 2 epochs
    if (e) % 2 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    
    # add the loss, comes in handy for plotting and checking that the generator and discriminator have the same learning speed
    discriminator_losses.append(discriminator_loss)
    generator_losses.append(generator_loss)

    ## code to check if images are made, no need to save these as they are easily made again when the model has been trained
    # if epoch % 5 == 0:
    #   noise = np.random.normal(0, 1, size=[100, latent_dim])
    #   generatedImages = generator.predict(noise)
    #   generatedImages = generatedImages.reshape(100, 96, 96, 3)          
    #   plotImages((generatedImages+1.0)/2.0, title='Epoch {}'.format(epoch))   
    #   saveModels(epoch)
  
    # save model every epoch
    gan.save_weights('gan_weights/gan.hdf5')
    generator.save_weights('gan_weights/generator.hdf5')
    discriminator.save_weights('gan_weights/discriminator.hdf5')


