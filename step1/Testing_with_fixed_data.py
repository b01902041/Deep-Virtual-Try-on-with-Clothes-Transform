from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from instance_normalization import InstanceNormalization
from keras.applications import *
import tensorflow as tf
import keras.backend as K
from keras.optimizers import RMSprop, SGD, Adam
import time
from IPython.display import clear_output
import cv2
from PIL import Image
import numpy as np
import glob
from random import randint, shuffle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

imageSize = 256
batchSize = 1
isRGB = True
apply_da = False
channel_axis = -1
channel_first = False


def load_data(file_pattern):
    return glob.glob(file_pattern)

def read_image(fn):
    input_size = (imageSize,imageSize)
    cropped_size = (imageSize,imageSize)
    
    if isRGB:
    # Load human picture
        im = Image.open(fn).convert('RGB')
        im = im.resize( input_size, Image.BILINEAR )    
    else:
        im = cv2.imread(fn)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        im = cv2.resize(im, input_size, interpolation=cv2.INTER_CUBIC)
    if apply_da is True:
        im = crop_img(im, input_size, cropped_size)
    arr = np.array(im)/255*2-1
    img_x_i = arr
    if channel_first:        
        img_x_i = np.moveaxis(img_x_i, 2, 0)
        
    # Load article picture y_i
    fn_y_i = fn[:-5] + "5.jpg"
    fn_y_i = fn_y_i[:fn_y_i.rfind("/")-1] + "5/" + fn_y_i.split("/")[-1]
    if isRGB:
        im = Image.open(fn_y_i).convert('RGB')
        im = im.resize(input_size, Image.BILINEAR )    
    else:
        im = cv2.imread(fn_y_i)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        im = cv2.resize(im, input_size, interpolation=cv2.INTER_CUBIC)
    arr = np.array(im)/255*2-1
    img_y_i = arr
    if channel_first:        
        img_y_i = np.moveaxis(img_y_i, 2, 0)
    
    # Load article picture y_j randomly
    fn_y_j = np.random.choice(filenames_5)
    while (fn_y_j == fn_y_i):
        fn_y_j = np.random.choice(filenames_5)
    if isRGB:
        im = Image.open(fn_y_j).convert('RGB')
        im = im.resize( input_size, Image.BILINEAR )
    else:
        im = cv2.imread(fn_y_j)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        im = cv2.resize(im, input_size, interpolation=cv2.INTER_CUBIC)
    arr = np.array(im)/255*2-1
    img_y_j = arr
    if randint(0,1): 
        img_y_j=img_y_j[:,::-1]
    if channel_first:        
        img_y_j = np.moveaxis(img_y_j, 2, 0)        
    
    if randint(0,1): # prevent disalign of the graphic on t-shirts and human when fplipping
        img_x_i=img_x_i[:,::-1]
        img_y_i=img_y_i[:,::-1]
    
    img = np.concatenate([img_x_i, img_y_i, img_y_j], axis=-1)    
    assert img.shape[-1] == 9
    
    return img

def minibatch_demo(data, batchsize, fn_y_i=None):
    length = len(data)
    epoch = i = 0
    tmpsize = None
    shuffle(data)
    while True:
        size = tmpsize if tmpsize else batchsize
        if i+size > length:
            shuffle(data)
            i = 0
            epoch+=1    
        rtn = [read_image(data[j]) for j in range(i,i+size)]
        i+=size
        tmpsize = yield epoch, np.float32(rtn)       

def minibatchAB_demo(dataA, batchsize, fn_y_i=None):
    batchA=minibatch_demo(dataA, batchsize, fn_y_i=fn_y_i)
    tmpsize = None    
    while True:        
        ep1, A = batchA.send(tmpsize)
        tmpsize = yield ep1, A

def showX(X, alpha, rows=1):
    assert X.shape[0]%rows == 0
    int_X = ((X+1)/2*255).clip(0,255).astype('uint8')
    int_alpha = ((alpha+1)/2*255).clip(0,255).astype('uint8')
    #print (int_X.shape)
    if channel_first:
        int_X = np.moveaxis(int_X.reshape(-1,3,imageSize,imageSize), 1, 3)
    else:
        if X.shape[-1] == 9:
            print("X.shape[-1] == 9")
            img_x_i = int_X[:,:,:,:3]
            img_y_i = int_X[:,:,:,3:6]
            img_y_j = int_X[:,:,:,6:9]
            int_X = np.concatenate([img_x_i, img_y_i, img_y_j], axis=1)
        else:
            
            print("X.shape[-1] != 9")
            print('int_X shape is', int_X.shape)
            #int_X = int_X.reshape(-1, 128, 128, 3)
            input_image = int_X[0, :, :, :].reshape(imageSize, imageSize, 3)
            target_image = int_X[2, :, :, :].reshape(imageSize, imageSize, 3)
            output_image = int_X[3, :, :, :].reshape(imageSize, imageSize, 3)
            #int_X = np.concatenate([img_x_i, img_y_i, img_y_j], axis=1)

    
    #int_X = int_X.reshape(rows, -1, 128, 128, 3).swapaxes(1,2).reshape(rows*imageSize, -1, 3)
    
    if not isRGB:
        int_X = cv2.cvtColor(int_X, cv2.COLOR_LAB2RGB)
    
    #print(int_alpha)
    print('int_alpha type is', int_alpha.dtype)
    print('int_alpha shape is', int_alpha.shape)
    int_alpha = int_alpha.reshape(imageSize, imageSize)
    int_alpha[int_alpha < 200] = 0
    int_alpha[int_alpha >= 200] = 255
    #int_alpha[:, 0:16] = 0
    #int_alpha[:, 111:128] = 0
    #print(int_alpha)
    
    Image.fromarray(input_image).save(os.path.join(origin_dir, filenames_5[idx].split("/")[-1]))
    Image.fromarray(output_image).save(os.path.join(output_dir, filenames_5[idx].split("/")[-1]))
    print(filenames_5[idx].split("/")[-1][:-4] + '_1.jpg')
    Image.fromarray(target_image).save(os.path.join(target_dir, filenames_5[idx].split("/")[-1][:-6] + '_1.jpg'))
    Image.fromarray(int_alpha.astype(np.uint8)).save(os.path.join(mask_dir, filenames_5[idx].split("/")[-1]))

def showG(A):
    def G(fn_generate, X):
        print('X.shape is', X.shape)
        print('X[0].shape is', X[0].shape)
        
        #r = np.array([fn_generate([X[i:i+1]]) for i in range(X.shape[0])])
        [fake_output, rec_input, alpha] = fn_generate([X[0:1]])
        #return r.swapaxes(0,1)[:,:,0]
        return fake_output, rec_input, alpha

    fake_output, rec_input, alpha = G(cycleA_generate, A)
    #_alpha = np.zeros((128, 128, 3), dtype = "uint8")
    #_alpha[:, :, 0] = alpha
    #_alpha[:, :, 1] = alpha
    #_alpha[:, :, 2] = alpha
    arr = np.concatenate([A[:,:,:,:3], A[:,:,:,3:6], A[:,:,:,6:9], fake_output, rec_input])

    showX(arr, alpha, 5)

def cycle_variables(netG1):
    """
    Intermidiate params:
        x_i: human w/ cloth i, shape=(128,96,3)
        y_i: stand alone cloth i, shape=(128,96,3)
        y_j: stand alone cloth j, shape=(128,96,3)
        alpha: mask for x_i_j, shape=(128,96,1)
        x_i_j: generated fake human swapping cloth i to j, shape=(128,96,3)
    
    Out:
        real_input: concat[x_i, y_i, y_j], shape=(128,96,9)
        fake_output: masked_x_i_j = alpha*x_i_j + (1-alpha)*x_i, shape=(128,96,3)
        rec_input: output of the second generator (generating image similar to x_i), shape=(128,96,3)
        fn_generate: a path from input to G_out and cyclic G_out
    """
    real_input = netG1.inputs[0]
    fake_output = netG1.outputs[0]
    # Legacy: how to split channels
    # https://github.com/fchollet/keras/issues/5474
    x_i = Lambda(lambda x: x[:,:,:, 0:3])(real_input)
    y_i = Lambda(lambda x: x[:,:,:, 3:6])(real_input)
    y_j = Lambda(lambda x: x[:,:,:, 6:])(real_input)
    alpha = Lambda(lambda x: x[:,:,:, 0:1])(fake_output)
    x_i_j = Lambda(lambda x: x[:,:,:, 1:])(fake_output)
    
    fake_output = alpha*x_i_j + (1-alpha)*x_i 
    concat_input_G2 = concatenate([fake_output, y_j, y_i], axis=-1) # swap y_i and y_j
    rec_input = netG1([concat_input_G2])
    rec_alpha = Lambda(lambda x: x[:,:,:, 0:1])(rec_input)
    rec_x_i_j = Lambda(lambda x: x[:,:,:, 1:])(rec_input)
    rec_input = rec_alpha*rec_x_i_j + (1-rec_alpha)*fake_output

    fn_generate = K.function([real_input], [fake_output, rec_input, alpha])
    return real_input, fake_output, rec_input, fn_generate, alpha

netGA = load_model('./models_512/netG1527659649.3044329.h5')
#netDA = load_model('./models/netD1526878347.025762.h5')

real_A, fake_B, rec_A, cycleA_generate, alpha_A = cycle_variables(netGA)

data = "MVC_image_pairs_resize_new"
train_A = load_data('./{}/1/*.jpg'.format(data))
filenames_5 = load_data('./{}/5_test/*.jpg'.format(data))

out_root_dir = "./testing_results_512_fixed_data"
origin_dir = out_root_dir + "/input_image"
target_dir = out_root_dir + "/target_image"
output_dir = out_root_dir + "/output_image"
mask_dir = out_root_dir + "/mask"

testing_number = 100
if not (os.path.exists(origin_dir)):
    os.makedirs(origin_dir)
if not (os.path.exists(target_dir)):
    os.makedirs(target_dir)
if not (os.path.exists(mask_dir)):
    os.makedirs(mask_dir)
if not (os.path.exists(output_dir)):
    os.makedirs(output_dir)

len_fn = len(filenames_5)
assert len_fn > 0


for n in range(testing_number):
    idx = np.random.randint(len_fn)
    fn = filenames_5[idx]
    demo_batch = minibatchAB_demo(train_A, batchSize, fn)
    epoch, A = next(demo_batch) 

    _, A = demo_batch.send(1)
    showG(A)


