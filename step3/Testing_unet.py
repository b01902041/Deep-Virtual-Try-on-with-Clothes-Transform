import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
from keras.applications.vgg16 import VGG16
from data import *
import glob
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


test_data_path = './Conditioning/testing_results_with_conditioning_data'
test_img_folder = 'target_image'
test_mask_folder = 'combined_mask'
model_name = './model_perceptualloss_256_conv5_mse_0.9999_0.0001/unet-41-0.26-0.19.hdf5'
result_dir = './Conditioning/testing_results_with_conditioning_data/warping_results'


def create_test_data():
    i = 0
    resize_size = 256
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    img_folder = os.path.join(test_data_path, test_img_folder)
    mask_folder = os.path.join(test_data_path, test_mask_folder)
    imgs_names = glob.glob(img_folder + "/*.jpg")

    imgdatas = np.ndarray((len(imgs_names), resize_size, resize_size, 3), dtype=np.uint8)
    imgmasks = np.ndarray((len(imgs_names), resize_size, resize_size, 1), dtype=np.uint8)

    for imgname in imgs_names:
        midname = imgname[imgname.rindex("/") + 1:]
        #print('midname is: ', midname)
        #maskname = midname.split("_")[0] + "_5.jpg"
        maskname = midname.split("_")[0]

        img = load_img(img_folder + "/" + midname, grayscale = False)
        
        img = img.resize( (resize_size, resize_size), Image.BILINEAR )

        img = img_to_array(img)

        img_mask = load_img(mask_folder + "/" + maskname, grayscale = True)

        img_mask = img_mask.resize( (resize_size, resize_size), Image.BILINEAR )

        img_mask = img_to_array(img_mask)

        imgdatas[i] = img
        imgmasks[i] = img_mask
        i += 1

    print('loading done')

    return imgdatas, imgmasks, imgs_names

def test(imgs_test, imgs_test_mask, model):
    print('predict test data')
    imgs_test_result = model.predict([imgs_test, imgs_test_mask], batch_size=1, verbose=1)
    return imgs_test_result

def get_model():
    resize_size = 256
    def perceptual_loss(y_true, y_pred): 
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(resize_size, resize_size, 3)) 
        loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output) 
        loss_model.trainable = False
        return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

    def perceptual_loss_multiple_layers(y_true, y_pred):
        #y_true = array_to_img(y_true)
        #y_pred = array_to_img(y_pred)
        #y_true = preprocess_input(y_true)
        #y_pred = preprocess_input(y_pred)
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(resize_size, resize_size, 3)) 
        loss_model_1 = Model(inputs=vgg.input, outputs=vgg.get_layer('block1_conv2').output) 
        loss_model_1.trainable = False

        loss_model_2 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv2').output) 
        loss_model_2.trainable = False

        loss_model_3 = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output) 
        loss_model_3.trainable = False

        loss_model_4 = Model(inputs=vgg.input, outputs=vgg.get_layer('block4_conv3').output) 
        loss_model_4.trainable = False

        loss_model_5 = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv3').output) 
        loss_model_5.trainable = False

        loss1 = K.mean(K.abs(loss_model_1(y_true) - loss_model_1(y_pred)))/1.6
        loss2 = K.mean(K.abs(loss_model_2(y_true) - loss_model_2(y_pred)))/2.3
        loss3 = K.mean(K.abs(loss_model_3(y_true) - loss_model_3(y_pred)))/1.8
        loss4 = K.mean(K.abs(loss_model_4(y_true) - loss_model_4(y_pred)))/2.8
        loss5 = K.mean(K.abs(loss_model_5(y_true) - loss_model_5(y_pred)))*10/0.8

        loss = loss1 + loss2 + loss3 + loss4 + loss5

        return loss

    def custom_loss_one_layer(y_true, y_pred): 
        weight_p = 0.9999
        weight_m = 0.0001
        #y_true_p = preprocess_input(y_true)
        #y_pred_p = preprocess_input(y_pred)
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(resize_size, resize_size, 3)) 
        
        loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv3').output) 
        loss_model.trainable = False

        loss_p = K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

        loss = weight_p * loss_p + weight_m * K.mean(K.square(y_pred - y_true), axis=-1)

        return loss

    def custom_loss_two_layers(y_true, y_pred): 
        weight_p = 0.9999
        weight_m = 0.0001
        #y_true_p = preprocess_input(y_true)
        #y_pred_p = preprocess_input(y_pred)
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(resize_size, resize_size, 3)) 
        
        loss_model_3 = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output) 
        loss_model_3.trainable = False

        loss_model_5 = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv3').output) 
        loss_model_5.trainable = False

        
        loss3 = K.mean(K.square(loss_model_3(y_true) - loss_model_3(y_pred)))
        loss5 = K.mean(K.square(loss_model_5(y_true) - loss_model_5(y_pred)))

        loss = weight_p * ((loss3 + loss5) / 2.0) + weight_m * K.mean(K.square(y_pred - y_true), axis=-1)

        return loss

    model = load_model(model_name, custom_objects={'custom_loss_one_layer': custom_loss_one_layer})
    return model

def save_img(imgs_names, imgs_test, imgs_test_mask, imgs_test_result):
    for i in range(imgs_test.shape[0]):
        img = imgs_test[i]
        #img = array_to_img(img)
        img = np.clip(img, 0, 255).astype('uint8')
        save_dir = os.path.join(result_dir, 'test_input')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        #img.save( os.path.join( save_dir, imgs_names[i].split('/')[-1] ) )
        imsave(os.path.join( save_dir, imgs_names[i].split('/')[-1] ), img)

    for i in range(imgs_test_mask.shape[0]):
        img = imgs_test_mask[i]
        img = array_to_img(img)
        save_dir = os.path.join(result_dir, 'test_mask')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img.save( os.path.join( save_dir, imgs_names[i].split('/')[-1] ) )

    for i in range(imgs_test_result.shape[0]):
        img = imgs_test_result[i]
        #img = array_to_img(img)
        img = np.clip(img, 0, 255).astype('uint8')
        save_dir = os.path.join(result_dir, 'test_result')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        #img.save( os.path.join( save_dir, imgs_names[i].split('/')[-1] ) )
        imsave(os.path.join( save_dir, imgs_names[i].split('/')[-1] ), img)

if __name__ == '__main__':
    imgs_test, imgs_test_mask, imgs_names = create_test_data()
    model = get_model()
    imgs_test_result = test(imgs_test, imgs_test_mask, model)
    save_img(imgs_names, imgs_test, imgs_test_mask, imgs_test_result)





