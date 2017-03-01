from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.datasets import mnist
import numpy as np
from keras import regularizers
from keras.utils.np_utils import to_categorical
from utils import *
import sys
sys.path.append('./libsvm-3.22/python')
from svmutil import *


input_shape = 64
encoding_dim = 500
sparsity = 1e-6


(x_train,y_train), (x_test,y_test) = mnist.load_data()
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

patch_num = 60000
patch_train = samplePatches(x_train, input_shape, patch_num)

x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))


model = Sequential()
model.add(Dense(encoding_dim,input_dim=input_shape,activation='relu',
	activity_regularizer=regularizers.activity_l1(sparsity)))
model.add(Dropout(0.5))
model.add(Dense(input_shape,activation='linear'))
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.fit(patch_train,patch_train,nb_epoch=30,batch_size=256,shuffle=True, validation_data=None)

w1 = model.layers[0].get_weights()[0]
b1 = model.layers[0].get_weights()[1]
w2 = model.layers[2].get_weights()[0]
b2 = model.layers[2].get_weights()[1]
feature_maps_train = createFM(x_train,w1,b1,'Max')
feature_maps_test = createFM(x_test,w1,b1,'Max')

svm_model = svm_train(y_train[:1000].tolist(),feature_maps_train[:1000].tolist(),'-c 5.5 -g 0.06')

p_label,p_acc,p_val = svm_predict(y_test[:1000].tolist(),feature_maps_test[:1000].tolist(),svm_model)
