import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
import tensorflow.keras.utils as np_utils
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


# Convert 1D class arrays to 10D class matrices
y_train = np_utils.to_categorical(y_train, 10)
y_val = np_utils.to_categorical(y_val, 10)
y_test = np_utils.to_categorical(y_test, 10)

## Exercise 2 ##

# Models
model1 = Sequential() # 0 hidden layers
model2 = Sequential() # 3 hidden layers with relu
model3 = Sequential() # 3 hidden layers with linear 

# Input layer
model1.add(Flatten(input_shape=(28,28,1)))
model2.add(Flatten(input_shape=(28,28,1)))
model3.add(Flatten(input_shape=(28,28,1)))

# 3 Hidden layers for model 2 and 3
model2.add(Dense(64, activation='relu'))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(64, activation='relu'))

model3.add(Dense(64, activation='linear'))
model3.add(Dense(64, activation='linear'))
model3.add(Dense(64, activation='linear'))

# Output layer
model1.add(Dense(10, activation='softmax'))
model2.add(Dense(10, activation='softmax'))
model3.add(Dense(10, activation='softmax'))

# Compile
model1.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model3.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Add a tensor board
board_1 = TensorBoard(r'logs\2_2_model1')
board_2 = TensorBoard(r'logs\2_2_model2')
board_3 = TensorBoard(r'logs\2_2_model3')

# Fit 
model1.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[board_1])
model2.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[board_2])
model3.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[board_3])

# Evaluation 
score_1 = model1.evaluate(X_test, y_test, verbose=0)
score_2 = model2.evaluate(X_test, y_test, verbose=0)
score_3 = model3.evaluate(X_test, y_test, verbose=0)

print("Loss: ",score_1[0])
print("Accuracy: ",score_1[1])

print("Loss: ",score_2[0])
print("Accuracy: ",score_2[1])

print("Loss: ",score_3[0])
print("Accuracy: ",score_3[1])