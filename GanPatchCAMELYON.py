import keras
import os
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

def Discriminator(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):

     model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(LeakyReLU(0.2))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, padding = 'same'))
     model.add(LeakyReLU(0.2))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Flatten())
     model.add(Dense(64))
     model.add(LeakyReLU(0.2))
     model.add(Dense(1, activation = 'sigmoid'))

     return model

def Generator():
  generator = keras.models.Sequential()
  generator.add(Dense(128*12*12, input_dim=latent_dim, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02)))
  generator.add(LeakyReLU(0.2))
  generator.add(Reshape((12,12,128)))
  generator.add(UpSampling2D(size=(2, 2)))
  generator.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
  generator.add(LeakyReLU(0.2))
  generator.add(UpSampling2D(size=(4, 4)))
  generator.add(Conv2D(3, kernel_size=(3, 3), padding='same', activation='tanh'))
  return generator

discriminator = Discriminator()
generator = Generator()

discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
generator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

# discriminator.summary()
# generator.summary()

discriminator.trainable = False
z = keras.layers.Input(shape=(latent_dim,))
x = generator(z)
D_G_z = discriminator(x)
gan = keras.models.Model(inputs=z, outputs=D_G_z)
gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

discriminator_losses = []
generator_losses = []

epochs = 2 
batch_size = 32

X_train = get_pcam_generators('C:\\Users\\Kirst\\Desktop\\TUe\\8P361-Project Imaging\\Project-Imaging-Group-3',train_batch_size=batch_size)

for e in range(epochs):
  print(f'Epoch {e}')
  for x_train,y_train in X_train:
    noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
    batch = x_train
    generated_images = generator.predict(noise)

    X = np.concatenate([batch, generated_images])

    labels_discriminator = np.zeros(2*batch_size)
    labels_discriminator[:batch_size] = 1
    print('.')
    discriminator.trainable = True
    discriminator_loss = discriminator.train_on_batch(X, labels_discriminator)

    noise = np.random.normal(0, 1, size=[batch_size, latent_dim])

    labels_generator = np.ones(batch_size)

    discriminator.trainable = False

    generator_loss = gan.train_on_batch(noise, labels_generator)

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

