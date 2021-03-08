# Goodfellow,  I.,  Bengio,  Y.,  and  Courville,  A.  (2016).Deep  Learning.MIT  Press.http://www.deeplearningbook.org

import keras
import os
import time
import numpy as np
import tensorflow as tf
import 
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

latent_dim = 100
IMAGE_SIZE = 96 

def plotImages(images, dim=(10, 10), figsize=(10, 10), title=''):
    plt.figure(figsize=figsize)
    for i in range(images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()


def get_pcam_generators(base_dir, train_batch_size=32):

     train_path = os.path.join(base_dir, 'train+val', 'train')
     
     RESCALING_FACTOR = 1./255

     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

    
     return train_gen

def Discriminator(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64,third_filters=32,fourth_filters=64):

     model = Sequential() 
     # (n,96,96,3)
     model.add(Conv2D(first_filters, kernel_size, padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3))) 
     # (n,96,96,32)
     model.add(LeakyReLU(0.2))
     # (n,96,96,32)
     model.add(MaxPool2D(pool_size = pool_size))
     #model.add(Conv2D(third_filters, kernel_size, strides=(4, 4), padding = 'same'))
     # (n,24,24,32)
     model.add(Conv2D(second_filters, kernel_size, padding = 'same'))
     # (n,24,24,64)
     model.add(LeakyReLU(0.2))
     # (n,24,24,64)
     model.add(MaxPool2D(pool_size = pool_size))
     #model.add(Conv2D(fourth_filters, kernel_size, strides=(4, 4), padding = 'same'))
     # (n,6,6,64)
     model.add(Flatten())
     # (n,2304)
     model.add(Dense(64))
     # (n,64)
     model.add(LeakyReLU(0.2))
     # (n,64)
     model.add(Dense(1, activation = 'sigmoid'))
     # (n,1)
     return model

def Generator():
  generator = keras.models.Sequential()
  # 
  generator.add(Dense(128*12*12, input_dim=latent_dim, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02)))
  # (n,18432)
  generator.add(LeakyReLU(0.2))
  # (n,18342)
  generator.add(Reshape((12,12,128)))
  # (n,12,12,128)
  generator.add(UpSampling2D(size=(2, 2)))
  # Conv2DTranspose can be used as well but is slower
  # (n,24,24,128)
  generator.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
  # (n,24,24,64)
  generator.add(LeakyReLU(0.2))
  # (n,24,24,64)
  generator.add(UpSampling2D(size=(4, 4)))
  # Conv2DTranspose can be used as well but is slower
  # (n,96,96,64)
  generator.add(Conv2D(3, kernel_size=(3, 3), padding='same', activation='tanh'))
  # (n,96,96,3)
  return generator

discriminator = Discriminator()
generator = Generator()

discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
generator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

# discriminator.summary()
# generator.summary()

discriminator.trainable = False
Input = keras.layers.Input(shape=(latent_dim,))
OutputGenerator = generator(Input)
OutputDiscriminator = discriminator(OutputGenerator)
gan = keras.models.Model(inputs=Input, outputs=OutputDiscriminator)
gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

discriminator_losses = []
generator_losses = []

epochs =  10
batch_size = 64

X_train = get_pcam_generators('C:\\Users\\Kirst\\Desktop\\TUe\\8P361-Project Imaging\\Project-Imaging-Group-3',train_batch_size=batch_size)
batches = 0

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator,discriminator=discriminator)
                                 
for epoch in range(epochs):

  print(f'Epoch {epoch}')
  start = time.time()

  for x_train,y_train in X_train:
    #  generate random input for the generator from normal distribition (z in Goodfellow et al. 2016)
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

    # generate ones for labels for generator
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

  # add the loss, comes in handy for plotting and checking that the generator and discriminator have the same learning speed
  generator_losses.append(generator_loss)

  
  if epoch % 5 == 0:
    noise = np.random.normal(0, 1, size=[100, latent_dim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(100, 96, 96, 3)          
    plotImages((generatedImages+1.0)/2.0, title='Epoch {}'.format(epoch))   
    saveModels(epoch)

