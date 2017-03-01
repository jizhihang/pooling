from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.datasets import mnist
import numpy as np
from keras import regularizers
from utils import *
import sys
sys.path.append('./libsvm-3.22/python')
from svmutil import *

# basic settings for autoencoder
input_shape = 64
encoding_dim = 500
sparsity = 1e-6

# load mnist dataset
(x_train,y_train), (x_test,y_test) = mnist.load_data()
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

# prepare patches for autoencoder training
patch_num = 60000
patch_train = samplePatches(x_train, input_shape, patch_num)

x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

# build an autoencoder and train
model = Sequential()
model.add(Dense(encoding_dim,input_dim=input_shape,activation='relu',
	activity_regularizer=regularizers.activity_l1(sparsity)))
model.add(Dropout(0.5))
model.add(Dense(input_shape,activation='linear'))
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.fit(patch_train,patch_train,nb_epoch=100,batch_size=256,shuffle=True, validation_data=None)

# get weights and bias
w1 = model.layers[0].get_weights()[0]
b1 = model.layers[0].get_weights()[1]
w2 = model.layers[2].get_weights()[0]
b2 = model.layers[2].get_weights()[1]

# create pooling feature maps at resolution 1
feature_maps_train = createFM(x_train,w1,b1,'Max')
feature_maps_test = createFM(x_test,w1,b1,'Max')

# train SVM and predict
svm_model = svm_train(y_train.tolist(),feature_maps_train.tolist(),'-c 5.5 -g 0.06')
p_label,p_acc,p_val = svm_predict(y_test.tolist(),feature_maps_test.tolist(),svm_model)
