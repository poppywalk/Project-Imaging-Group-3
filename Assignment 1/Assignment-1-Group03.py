## Exercise 4 - Group 03##

import os
import matplotlib.pyplot as plt
import random

# Amount of pictures taken from each folder (the 0 folder and the 1 folder)
n = 4

# Path for the benign images
trainzero   = 'C:\\Users\\Kirst\\Desktop\\TUe\\8P361-Project Imaging\\Project-Imaging-Group-3\\train+val\\train\\0'

# Path for the malignent images
trainone    = 'C:\\Users\\Kirst\\Desktop\\TUe\\8P361-Project Imaging\\Project-Imaging-Group-3\\train+val\\train\\1'

# All images with label 0
zero_images = os.listdir(trainzero)
# All images with label 1
one_images  = os.listdir(trainone)

# Make the figures and plot in subplots
fig, axes = plt.subplots(2, n)

# Take a random image out of both folders and show what class they belong to and which images it is out of the folder
for i in range(0,n):
    # Select a random image
    index_zero  = random.randint(0,len(zero_images))
    # Take the random image from the 0 folder
    select_zero = plt.imread(trainzero + '\\' + zero_images[index_zero])
    # Select a random image
    index_one   = random.randint(0,len(one_images))
    # Take the random image from the 1 folder
    select_one  = plt.imread(trainone + '\\' + one_images[index_one])
    # Show the images with the correct title 
    axes[0,i].imshow(select_zero)
    axes[1,i].imshow(select_one)
    axes[0,i].set_title(f'Class zero, image: {index_zero}')
    axes[1,i].set_title(f'Class one, image: {index_one}')

# Show the images in the plot
plt.show()

    
    