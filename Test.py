import tensorflow as tf 
dev  = tf.config.experimental.list_physical_devices('GPU')
CUDA = tf.test.is_built_with_cuda()
if dev and CUDA: print('succes')
