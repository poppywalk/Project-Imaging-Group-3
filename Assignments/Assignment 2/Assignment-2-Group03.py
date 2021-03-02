# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

# import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

# load the dataset using the builtin Keras method
(X_train, y_train), (X_test, y_test) = mnist.load_data()


###### --------------------------------------------------------------------------- ######

## Exercise 3 - Group 03 ##

###### Labeling for a four class classification problem with the help of a dictionairy ######

#labeling = {0:1,1:0,2:2,3:3,4:3,5:2,6:1,7:0,8:1,9:1}

#def Make_new_array(old_array):
#    new_array = np.zeros(old_array.shape)
#    for n in range(old_array.shape[0]):
#        label = old_array[n]
#        newlabel = labeling[label]
#        new_array[n] = newlabel
#    return new_array


#y_train_4 = Make_new_array(y_train)
#y_val_4 = Make_new_array(y_val)
#y_test_4 = Make_new_array(y_test)

###### Labeling for a four class classification problem with the help of for loops ######

def labeling(array):
    #Rewrite the classes from 10D to 4D with the following rules 
    #"vertical digits" 1 and 7 are class 0
    #"Loopy digits" 0,6,8,9 are class 1
    #"Curly digits" 2,5 are class 2
    #"Other" 3,4 are class 3
    array_new = array
    for i in [1,7]:
        array_new = np.where(array==i, 0, array_new)
    for i in [0,6,8,9]:
        array_new = np.where(array==i, 1, array_new)
    for i in [2,5]:
        array_new = np.where(array==i, 2, array_new)
    for i in [3,4]:
        array_new = np.where(array==i, 3, array_new)
    array = array_new
    return array
y_train = labeling(y_train)
y_test = labeling(y_test)

###### --------------------------------------------------------------------------- ######

# derive a validation set from the training set
# the original training set is split into 
# new training set (90%) and a validation set (10%)
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)

# the shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxC, where
# C is the channel dimension
X_train = np.reshape(X_train, (-1,28,28,1)) 
X_val = np.reshape(X_val, (-1,28,28,1))
X_test = np.reshape(X_test, (-1,28,28,1))

# convert the datatype to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

# normalize our data values to the range [0,1]
X_train /= 255
X_val /= 255
X_test /= 255

# convert 1D class arrays to N_D class matrices
N_D = 4 #Amount of dimensions
y_train = to_categorical(y_train, N_D)
y_val = to_categorical(y_val, N_D)
y_test = to_categorical(y_test, N_D)


###### When using the dictionairy for exercise 3 use the lines below, otherwise the lines above ######

# Convert 1D class arrays to 10D class matrices
#y_train = to_categorical(y_train_4, 4)
#y_val = to_categorical(y_val_4, 4)
#y_test = to_categorical(y_test_4, 4)

###### ---------------------------------------------------------------------------------------- ######

model = Sequential()

# flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
model.add(Flatten(input_shape=(28,28,1))) 


###### --------------------------------------------------------------------------------######

## Exercise 1 and exercise 2- Group 03 ##
# Change the code below to add more layers or to change the ReLu to linear

# Amount of neurons per layer. When a change in neurons per layer is needed change N_neuron to the desired amount of neurons
N_neuron = 64
# When linearity is needed, change activation='relu' to activation='linear'
model.add(Dense(N_neuron, activation='relu'))
model.add(Dense(N_neuron, activation='relu'))
model.add(Dense(N_neuron, activation='relu'))
# Comment out the next line when three layers are needed.
# model.add(Dense(N_neuron, activation='relu'))
# When a new layer is needed, just add: model.add(Dense(N_neuron, activation='relu')) per layer, when a relu is used (otherwise change relu)

###### --------------------------------------------------------------------------------######

# output layer with 10 nodes (one for each class) and softmax nonlinearity
model.add(Dense(N_D, activation='softmax')) 

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# use this variable to name your model
model_name="test"

# create a way to monitor our model in Tensorboard
tensorboard = TensorBoard("logs/" + model_name)

# train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])

score = model.evaluate(X_test, y_test, verbose=0)

print("Loss: ",score[0])
print("Accuracy: ",score[1])
