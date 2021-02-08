import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# print('Training done on CPU')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard


# Load the dataset using the builtin Keras method
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Derive a validation set from the training set
# The original training set is split into 
# new training set (90%) and a validation set (10%)
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)



# The shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxH, where
# C is the channel dimension
X_train = np.reshape(X_train, (-1,28,28,1)) 
X_val = np.reshape(X_val, (-1,28,28,1))
X_test = np.reshape(X_test, (-1,28,28,1))


# Convert the datatype to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')


# Normalize our data values to the range [0,1]
X_train /= 255
X_val /= 255
X_test /= 255


## Exercise 3 ##

# Creating new class arrays for four-class problem

labeling = {0:1,1:0,2:2,3:3,4:3,5:2,6:1,7:0,8:1,9:1}

def Make_new_array(old_array):
    new_array = np.zeros(old_array.shape)
    for n in range(old_array.shape[0]):
        label = old_array[n]
        newlabel = labeling[label]
        new_array[n] = newlabel
    return new_array

y_train_4 = Make_new_array(y_train)
y_val_4 = Make_new_array(y_val)
y_test_4 = Make_new_array(y_test)
print(y_train_4)

# Convert 1D class arrays to 10D class matrices
y_train = to_categorical(y_train_4, 4)
y_val = to_categorical(y_val_4, 4)
y_test = to_categorical(y_test_4, 4)
print(y_train)

# Model
model = Sequential()

model.add(Flatten(input_shape=(28,28,1)))

# 3 Hidden layers
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

# Output layer
model.add(Dense(4, activation='softmax'))

# Compile
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Add a tensor board
model_name = "Custom_model_1_4class"
board = TensorBoard(r'logs\ ' + model_name)

# Fit 
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[board])

# Print score
score = model.evaluate(X_test, y_test, verbose=0)

print("Loss: ",score[0])
print("Accuracy: ",score[1])