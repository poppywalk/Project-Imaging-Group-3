import os
import matplotlib.pyplot as plt
import random

n = 4

trainzero   = 'C:\\Users\\Kirst\\Desktop\\TUe\\8P361-Project Imaging\\Project-Imaging-Group-3\\train+val\\train\\0'
trainone    = 'C:\\Users\\Kirst\\Desktop\\TUe\\8P361-Project Imaging\\Project-Imaging-Group-3\\train+val\\train\\1'
zero_images = os.listdir(trainzero)
one_images  = os.listdir(trainone)

fig, axes = plt.subplots(2, n)

for i in range(0,n):
    index_zero  = random.randint(0,len(zero_images))
    select_zero = plt.imread(trainzero + '\\' + zero_images[index_zero])
    index_one   = random.randint(0,len(one_images))
    select_one  = plt.imread(trainone + '\\' + one_images[index_one])
    axes[0,i].imshow(select_zero)
    axes[1,i].imshow(select_one)
    axes[0,i].set_title(f'Class zero, image: {index_zero}')
    axes[1,i].set_title(f'Class one, image: {index_one}')

plt.show()

    
    