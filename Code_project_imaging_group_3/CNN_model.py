# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):
    """
    Function that defines the training data

    Parameters: base_dir: string in which the path where the train folder is located can be assigned
              train_batch_size: integer which shows the batch size of training data, default is 32

    Returns: train_gen: directoryiterater of the training data
             val_gen: directoryiterater of the validation data
    """
     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val', 'train')
     valid_path = os.path.join(base_dir, 'train+val', 'valid')


     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary',shuffle=False)

     return train_gen, val_gen

def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):
    """
    Defines the CNN the model
    Inputs: kernel_size: tuple which shows the size of the convolutional kernel, default is (3,3)
          pool_size: integer which shows the size of the max pooling kernel, default is (4,4)
          first/second_filters: integers which show the amount of neurons, default is 32,64,32,64 respectively
    Return: A keras sequential model of the CNN_model
    """
     # build the model
     model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Flatten())
     model.add(Dense(64, activation = 'relu'))
     model.add(Dense(1, activation = 'sigmoid'))

     # compile the model
     model.compile(SGD(lr=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model


# get the model
model = get_model()

# Check the layers
model.summary()

# get the data generators
train_gen, val_gen = get_pcam_generators('C:\\Users\\Kirst\\Desktop\\TUe\\8P361-Project Imaging\\Project-Imaging-Group-3')


# save the model and weights
model_name = 'Fifth_run_model'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)

# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

history = model.fit(train_gen, steps_per_epoch=train_steps,
                     validation_data=val_gen,
                     validation_steps=val_steps,
                     epochs=3,
                     callbacks=callbacks_list)


# ROC analysis
model.load_weights('Fifth_run_model_weights.hdf5')
# Get the predictions of the model
predictions = model.predict(val_gen)
# Plot a line from 0 to 1 with stepsize 0.01
line = np.arange(start=0,stop=1,step=0.01)
# Make the roc curve with the help of the predictions of the model
fpr, tpr,_ = roc_curve(val_gen.classes,predictions)
# Plot the figure
plt.plot(line,line,'--',fpr,tpr,'-')
# Give the figure a title
plt.title('ROC-curve')
# Give the x-axis a label
plt.xlabel('False Positive Rate')
# Give the y-axis a label
plt.ylabel('True Positive Rate')
# Calculate the AUC
AUC = auc(fpr,tpr)
# Print the AUC in the plot
plt.text(0.8,0.2,f'AUC = {AUC:.3f}')
# Print the AUC
print(AUC)
# Show the plot
plt.show()
