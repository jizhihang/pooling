import numpy as np
import random
import math
import scipy
from scipy import signal

def samplePatches(input_im, input_shape, patch_num):
    """
    sample patches from input images
    input_shape should be a square number
    """

    num_imgs = input_im.shape[0]
    width = input_im.shape[1]
    height = input_im.shape[2]
    patches = np.zeros((patch_num,input_shape))
    side = int(math.sqrt(input_shape))
    for i in range(patch_num):
      im_idx = random.randint(0,num_imgs-1)
      start_w = random.randint(0,width-side)
      start_h = random.randint(0,height-side)
      patch = input_im[im_idx,start_w:start_w+side,start_h:start_h+side]
      patches[i,:] = patch.reshape((1,input_shape))

    return patches

def ReLU(ori_map):
    """
    relu activation
    original map should be a 2d np array
    """
    ori_map[ori_map<0] = 0
    return ori_map

def createFM(imgs, weights, bias, TYPE):
    """
    create feature maps with weights and bias from encoder
    convolve imgs with filters reshaped from weights, add bias and do activation
    TYPE indicates type of pooling to be used, supporting Max, Average
    """
    num_imgs = imgs.shape[0]
    im_side = int(math.sqrt(imgs.shape[1]))
    num_filt = weights.shape[1]
    weight_len = weights.shape[0]
    weight_len = int(math.sqrt(weight_len))
    
    feature_maps = np.zeros((num_imgs,num_filt)).astype('float32')
    for k in range(num_imgs):
      img = imgs[k,:].reshape(im_side,im_side)
      for i in range(num_filt):
        filt = weights[:,i].reshape((weight_len,weight_len))
        feature_map = scipy.signal.convolve(img,filt,mode='valid')
        feature_map = feature_map + bias[i]
        feature_map = ReLU(feature_map)
        if TYPE=='Max':
	  feature_maps[k,i] = feature_map.max()
	elif TYPE=='Average':
	  feature_maps[k,i] = feature_map.mean()
        else:
	  print "Unsupported pooling type..."
	  break

    return feature_maps



