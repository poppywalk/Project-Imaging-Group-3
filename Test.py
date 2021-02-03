import matplotlib.pyplot as plt



def testshow(path):
    img = plt.imread(path)
    plt.imshow(img)
    plt.show()
    return


train0 = ('0000d563d5cfafc4e68acb7c9829258a298d9b6a.tif','19ce0d074244746c7ea328cc1842e9211f1b6f01.tif','19d2e5774cd4741e16b40cf6e4266b3e3cfd6705.tif')
train1 = ('0000da768d06b879e5754c43e2298ce48726f722.tif','000aa5d8f68dc1f45ebba53b8f159aae80e06072.tif','00a41ae9577f0df441b8c2794b184f48cd561289.tif')
path0 ='C:\\Users\\Kirst\\Desktop\\TUe\\8P361-Project Imaging\\Project-Imaging-Group-3\\train+val\\train\\0\\'
path1 ='C:\\Users\\Kirst\\Desktop\\TUe\\8P361-Project Imaging\\Project-Imaging-Group-3\\train+val\\train\\1\\'


k=0
for j in train0:
    paths = path0+train0[k]
    k = k+1
    testshow(paths)

    