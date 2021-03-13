# Goodfellow,  I.,  Bengio,  Y.,  and  Courville,  A.  (2016).Deep  Learning.MIT  Press.http://www.deeplearningbook.org
''' Module which makes a simple Gan, to generate new images from the dataset given
'''

import keras
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Input, Embedding, LeakyReLU, Reshape, Concatenate, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model

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
epochs =  1
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


def Generator(latent_dim=100):
  """
  The Generator is a neural network that generates fake images

  Inputs: latent_dim = integer, which shows the size of the latent dimension. 
          the latent dimension is the space with possible inputs for the generator. 
          every vector in the latent dimension gives a certain image which the generator can make.
          default is 100, which means it is 100 dimensional, more dimensions would give more complexity.

  Returns: keras sequential model of the generator model
  """
  # first input for the images
  # input shape as big as the latent dimension (n,100)
  images_input = Input(shape = (latent_dim,))
  # add Dense layer for the input shape (n,18432)
  images_dense = Dense(128*12*12, input_dim=latent_dim, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(images_input)
  # add a LeakyReLU layer (n,18432)
  images_leakyrelu = LeakyReLU(alpha=0.2)(images_dense)
  # add a Reshape layer to get the wanted output structure (n,12,12,128)
  images_reshape = Reshape((12,12,128))(images_leakyrelu)

  # second input for the labels
  # input shape for the labels (an array) (n,1)
  labels_input = Input(shape=(1,))
  # add an Embedding layer, converts the label to a latent space vector (n,1,50)
  labels_embedding = Embedding(2, 50)(labels_input)
  # add a Dense layer (n,1,432)
  labels_dense = Dense(12*12*3)(labels_embedding)
  # add a Reshape layer to convert to the wanted structure (n,12,12,3)
  labels_reshape = Reshape((12,12,3))(labels_dense)

  # merge the two models into one
  merge = Concatenate()([images_reshape, labels_reshape])
  
  # upsample to (24,24)
  # add a Conv2DTranspose to upsample (n,24,24,128)
  upsample1 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
  # add a Leaky ReLU layer to activate it (Leaky ReLU is not a standard activation) (n,24,24,128)
  leakyrelu1 = LeakyReLU(alpha=0.2)(upsample1)

  # upsample to (48,48)
  # add a Conv2DTranspose to upsample (n,48,48,128)
  upsample2 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(leakyrelu1)
  # add a Leaky ReLU layer to activate it (Leaky ReLU is not a standard activation) (n,48,48,128)
  leakyrelu2 = LeakyReLU(alpha=0.2)(upsample2)

  # upsample to (96,96)
  # add a Conv2DTranspose to upsample (n,96,96,128)
  upsample3 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(leakyrelu2)
  # add a Leaky ReLU layer to activate it (Leaky ReLU is not a standard activation) (n,96,96,128)
  leakyrelu3 = LeakyReLU(alpha=0.2)(upsample3)

  # output
  # add a convolutional layer for the output (n,96,96,3)
  out_layer = Conv2D(3, (3, 3), padding='same', activation='tanh')(leakyrelu3)

  # make the model
  generator = Model([images_input, labels_input], out_layer)

  # compile the model
  generator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())

  return generator

generator = Generator()
generator.summary()

def Discriminator(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64,third_filters=32,fourth_filters=64):
  """
  The discriminator is a Neural network that determines if the generated images are real or fake thereby updating the generator.

  Inputs: kernel_size: tuple which shows the size of the convolutional kernel, default is (3,3)
          pool_size: integer which shows the size of the max pooling kernel, default is (4,4)
          first/second/third/fourth_filters: integers which show the amount of neurons, default is 32,64,32,64 respectively
          negativeslopecoefficient: float which shows the value of the slope coefficient for the Leaky ReLU layer, default is 0.2

  Returns: keras sequential model of the discriminator model
  """

  # input shape for the labels (an array) (n,1)     
  labels_input = Input(shape=(1,))
  # add an Embedding layer, converts the label to a latent space vector (n,1,50)
  labels_embedding = Embedding(2, 50)(labels_input)
  # add a Dense layer (n,1,27648)
  labels_dense = Dense(96*96*3)(labels_embedding)
  # add a Reshape layer to convert to the wanted structure (n,96,96,3)
  labels_reshape = Reshape((96,96,3))(labels_dense)

  # define the input shape (n,96,96,3)
  images_input = Input(shape=(96,96,3))

  # merge the images and the labels 
  merge = Concatenate()([images_input, labels_reshape])

  # add a convolutional layer (input =(n,96,96,3))
  First_layer = Conv2D(first_filters, kernel_size, padding = 'same')(merge) 
  # add a Leaky ReLU layer to activate it (Leaky ReLU is not a standard activation) (input =(n,96,96,32))
  First_Leaky_relu= LeakyReLU(0.2)(First_layer)
  # (input =(n,96,96,32))
  # the max pooling or strided convolutional layer scales the images down with a factor of 4
  First_Max_Pool = MaxPool2D(pool_size = pool_size)(First_Leaky_relu)
  # add a convolutional layer (input =(n,24,24,32))
  Second_layer = Conv2D(first_filters, kernel_size, padding = 'same')(First_Max_Pool) 
  # add a Leaky ReLU layer to activate it (Leaky ReLU is not a standard activation) (input =(n,24,24,32))
  Second_Leaky_relu= LeakyReLU(0.2)(Second_layer)
  # the max pooling or strided convolutional layer scales the images down with a factor of 4 (input =(n,24,24,32))
  Second_Max_Pool = MaxPool2D(pool_size = pool_size)(Second_Leaky_relu)
  # flatten the output of the Max Pooling layer (input =(n,6,6,64))
  Flatten = keras.layers.Flatten()(Second_Max_Pool)
  # add a Dense layer (input =(n,2304))
  Dense_1 = Dense(64)(Flatten)
  # add a Leaky ReLU layer to activate it (Leaky ReLU is not a standard activation) (input =(n,64))
  Third_Leaky_relu = LeakyReLU(0.2)(Dense_1)
  # add Dense layer for the output (input = (n,64))
  output1 = Dense(1, activation = 'sigmoid')(Third_Leaky_relu)
  # (output =(n,1))

  # make the model
  discriminator = Model(inputs=[images_input, labels_input],outputs=output1)

  # compile the model
  discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())

  return discriminator

discriminator = Discriminator()
discriminator.summary()

def Get_Gan(discriminator, generator,latent_dim=latent_dim):
  """
  Function that gets the gan

  Parameters: latent_dim: integer which shows the latent dimensions
              discriminator: keras sequential model 
              generator: keras sequential model

  Returns: gan: model
  """
  discriminator.trainable = False
  gen_noise, gen_labels = generator.input
  
  OutputGenerator = generator.output
  # use both outputs of the discriminator
  OutputDiscriminator = discriminator([OutputGenerator, gen_labels])
  # use both outputs of the generator in the GAN
  gan = keras.models.Model(inputs=[gen_noise, gen_labels], outputs=OutputDiscriminator)
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
  gan = Get_Gan(discriminator, generator, latent_dim)  

  for epoch in range(epochs):

    print(f'Epoch {epoch+1}')
    start = time.time()
    batches = 0 
    for x_train,y_train in X_train:
      # generate random input for the generator from normal distribition (z in Goodfellow et al. 2016)
      # batch_size is the amount of images we want to have
      geninput = np.random.normal(0, 1, size=[batch_size, latent_dim])
      # half of the images are made to be 0 half are 1
      genlabels = np.zeros((batch_size,))
      genlabels[:batch_size//2] = 1
      np.random.shuffle(genlabels)
      # use the generator to generate new random images from the generator input
      generated_images = generator.predict([geninput,genlabels])

      # concatenate the real images with the generated images, dataset with the real images followed by the fake images 
      X = np.concatenate([x_train, generated_images])
      X_labels = np.concatenate([y_train, genlabels])

      # label the dataset, so 2 times labels of size batch_size for the generated images and the real ones
      labels_discriminator = np.zeros(2*batch_size)

      # label the real images with 1 and leave the fake to 0
      labels_discriminator[:batch_size] = 1
      
      # set the discriminator weights to trainable, so it can be trained
      discriminator.trainable = True

      # train the discriminator, and output the value of the loss function
      discriminator_loss = discriminator.train_on_batch([X,X_labels], labels_discriminator)

      #  generate new input for the generator from normal distribition (z in Goodfellow et al. 2016)
      geninput = np.random.normal(0, 1, size=[batch_size, latent_dim])
      genlabels = np.zeros((batch_size,))
      genlabels[:batch_size//2] = 1
      np.random.shuffle(genlabels)

      # generate ones for labels for generator for the loss function
      labels_generator = np.ones(batch_size)

      # fix the discriminator weights before training the generator, otherwise the discriminator will keep on trying to train
      discriminator.trainable = False

      # train the generator, and output the value of the loss function
      generator_loss = gan.train_on_batch([geninput,genlabels], labels_generator)
      
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


Train_Gan(epochs, X_train, batch_size, latent_dim, discriminator, generator)




