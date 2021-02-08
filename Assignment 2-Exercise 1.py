import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
import tensorflow.keras.utils as np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard


# load the dataset using the builtin Keras method
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# derive a validation set from the training set
# the original training set is split into 
# new training set (90%) and a validation set (10%)
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)



# the shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxH, where
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


# convert 1D class arrays to 10D class matrices
y_train = np_utils.to_categorical(y_train, 10)
y_val = np_utils.to_categorical(y_val, 10)
y_test = np_utils.to_categorical(y_test, 10)

## Exercise 1 ##

# Four layer model
model_4 = Sequential()

# Input layer
model_4.add(Flatten(input_shape=(28,28,1)))

# 4 Hidden layers
model_4.add(Dense(64, activation='relu'))
model_4.add(Dense(64, activation='relu'))
model_4.add(Dense(64, activation='relu'))
model_4.add(Dense(128, activation='relu'))

# Output layer
model_4.add(Dense(10, activation='softmax'))

# Compile
model_4.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Add a monitoring board
model_4_title = "Four-layer-model"
board = TensorBoard(r'logs\ ' + model_4_title)

# Fit the data
model_4.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[board])

# Obtain evaluation and print
score_4 = model_4.evaluate(X_test, y_test, verbose=0)

print("Loss: ",score_4[0])
print("Accuracy: ",score_4[1])