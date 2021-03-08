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

def RunOnGPU():
  """
  Function that makes it so that tensorflow uses the GPU for computing
  parameters:none
  returns:none
  """
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  #Limit memory of the GPU to 2 GB
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

#define latent dimension and the image dimensions
latent_dim = 100
IMAGE_SIZE = 96 

def plotImages(images, dim=(10, 10), figsize=(10, 10), title=''):
  """
  Function that plots images
  Parameters: images, a collection of images,
          dim, the dimensions of 
          figsize, the figure size
          title, the title of the generated plot
  returns: A plot of the images
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
  Function that defines the training and validation data
  parameters: base_dir, the path where the train and validation folders are located
              train_batch_size, The batch size of training data
  Returns
  """
  #Defines the path for the training and validation set
  train_path = os.path.join(base_dir, 'train+val', 'train')
  
  #Rescale the grey values to be between 0 and 1
  RESCALING_FACTOR = 1./255

  datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

  train_gen = datagen.flow_from_directory(train_path,
                                          target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                          batch_size=train_batch_size,
                                          class_mode='binary')
  return train_gen

def Discriminator(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64,third_filters=32,fourth_filters=64):
  """
  The discriminator is a Neural network that determines if the generated images are real or fake thereby updating the generator.
  Inputs: kernel_size: The size of the convolutional kernel
          pool_size: The size of the max pooling kernel
          first/second/third/fourth_filters:
  Returns: discriminator, the discriminator model
  """
  input_layer = keras.layers.Input(shape = (IMAGE_SIZE,IMAGE_SIZE,3))
  # (n,96,96,3)
  First_layer = Conv2D(first_filters, kernel_size, padding = 'same')(input_layer) 
  # (n,96,96,32)
  First_Leaky_relu= LeakyReLU(0.2)(First_layer)
  # (n,96,96,32)
  #The max pooling or strided convolutional layer scales the images down with a factor of 4
  First_Max_Pool = MaxPool2D(pool_size = pool_size)(First_Leaky_relu)
  #model.add(Conv2D(third_filters, kernel_size, strides=(4, 4), padding = 'same'))
  # (n,24,24,32)
  Second_layer = Conv2D(first_filters, kernel_size, padding = 'same')(First_Max_Pool) 
  # (n,24,24,32)
  Second_Leaky_relu= LeakyReLU(0.2)(Second_layer)
  # (n,24,24,32)
  #The max pooling or strided convolutional layer scales the images down with a factor of 4
  Second_Max_Pool = MaxPool2D(pool_size = pool_size)(Second_Leaky_relu)
  #model.add(Conv2D(fourth_filters, kernel_size, strides=(4, 4), padding = 'same'))
  # (n,6,6,64)
  Flatten = keras.layers.Flatten()(Second_Max_Pool)
  # (n,2304)
  Dense_1 = Dense(64)(Flatten)
  # (n,64)
  Third_Leaky_relu = LeakyReLU(0.2)(Dense_1)
  # (n,64)
  output1 = Dense(1, activation = 'sigmoid')(Third_Leaky_relu)
  # (n,1)
  discriminator = keras.Model(inputs=input_layer,outputs=output1)
  #Compile the model by specifying the loss and optimizer
  discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
  return discriminator

def Generator():
  """
  The Generator is a neural network that generates fake images
    Inputs: none
    Returns: generator, the generator model
  """
  generator = keras.models.Sequential()
  # 
  generator.add(Dense(128*12*12, input_dim=latent_dim, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02)))
  # (n,18432)
  generator.add(LeakyReLU(0.2))
  # (n,18342)
  generator.add(Reshape((12,12,128)))
  # (n,12,12,128)
  #Upsampeling, upsamples the image by a factor of two
  generator.add(UpSampling2D(size=(2, 2)))
  # Conv2DTranspose can be used as well but is slower
  # (n,24,24,128)
  generator.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
  # (n,24,24,64)
  generator.add(LeakyReLU(0.2))
  # (n,24,24,64)
  #Upsampeling, upsamples the image by factor of four
  generator.add(UpSampling2D(size=(4, 4)))
  # Conv2DTranspose can be used as well but is slower
  # (n,96,96,64)
  generator.add(Conv2D(3, kernel_size=(3, 3), padding='same', activation='tanh'))
  # (n,96,96,3)
  #Compile the model by specifying the loss and optimizer
  generator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
  return generator


def Gan(latent_dim=100):
  """
  Function that defines the full gan model by combining the generator and discriminator
  parameters: latent_dim
  Returns: gan, the gan model
  """
  #Ik snap even niet wat het nut is van deze code
  discriminator = Discriminator()
  generator = Generator()
  # discriminator.summary()
  # generator.summary()

  discriminator.trainable = False
  z = keras.layers.Input(shape=(latent_dim,))
  x = generator(z)
  D = discriminator(x)
  gan = keras.models.Model(inputs=z, outputs=D)
  gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
  return gan
# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#functies om toe te voegen 
#Generate real samples
#Generate fake samples
#Summarize performance, een functie die plots opslaat en het model opslaat

def train(generator,discriminator,gan,X_train,latent_dim,epochs,batch_size):
  """
  Function that trains the model
  Parameters: generator, the generator model
              discriminator, the discriminator model
              gan, the gan model
              X_train, the training dataset
              latent_dim, 
              N_epochs, the amount of epochs the model will train for
              batch_size, the amount of images per batch
  Returns
  """
  discriminator_losses = []
  generator_losses = []
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator=generator,discriminator=discriminator)
  batches = 0
  for e in range(epochs):
    print(f'Epoch {e}')
    start = time.time()
    for x_train,y_train in X_train:
      
      noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
      batch = x_train
      generated_images = generator.predict(noise)

      X = np.concatenate([batch, generated_images])

      labels_discriminator = np.zeros(2*batch_size)
      labels_discriminator[:batch_size] = 1
      
      discriminator.trainable = True
      discriminator_loss = discriminator.train_on_batch(X, labels_discriminator)

      noise = np.random.normal(0, 1, size=[batch_size, latent_dim])

      labels_generator = np.ones(batch_size)

      discriminator.trainable = False

      generator_loss = gan.train_on_batch(noise, labels_generator)
      
      batches += 1
      print(batches)
      if batches >= 144000 / batch_size:
        break
    print ('Time for epoch {} is {} sec'.format(e, time.time()-start))
    if (e) % 2 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    discriminator_losses.append(discriminator_loss)
    generator_losses.append(generator_loss)
    # if e % 5 == 0:
    #   noise = np.random.normal(0, 1, size=[100, latent_dim])
    #   generatedImages = generator.predict(noise)
    #   generatedImages = generatedImages.reshape(100, 96, 96, 3)          
    #   plotImages((generatedImages+1.0)/2.0, title='Epoch {}'.format(e))
    #   display.display(plt.gcf())
    #   display.clear_output(wait=True)
    #   time.sleep(0.001)    
    #   saveModels(e)

#Use the RunOnGPU function to run the training process on the GPU
# RunOnGPU()
#Loading in the training data
X_train = get_pcam_generators(r"C:\Users\20174069\Desktop\Project imaging",train_batch_size=64)


#training the model
train(generator=Generator(),discriminator=Discriminator(),gan=Gan(),X_train=X_train,latent_dim=latent_dim, epochs =  1, batch_size = 64)