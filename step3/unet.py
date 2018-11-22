import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from data import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


class myUnet(object):
    

    def __init__(self, img_rows = 641, img_cols = 641):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.resize_size = 256

    def load_data(self):

        mydata = dataProcess(self.img_rows, self.img_cols)
        #mydata = dataProcess(self.img_rows, self.img_cols, data_path = "./MVC_for_debug/shirts_1", mask_path = "./MVC_for_debug/fc8_mask_5_modified", label_path = "./MVC_for_debug/shirts_5")
        #imgs_train, imgs_train_mask, imgs_train_label = mydata.load_train_data()
        imgs_train, imgs_train_mask, imgs_train_label = mydata.create_train_data()
        #imgs_test, imgs_test_mask = mydata.load_test_data()
        imgs_test, imgs_test_mask = mydata.create_test_data()

        return imgs_train, imgs_train_mask, imgs_train_label, imgs_test, imgs_test_mask


    def get_unet(self):


        def perceptual_loss(y_true, y_pred): 
            #y_true *= 255
            #y_pred *= 255

            #y_true = K.reverse(y_true, 3)
            #y_pred = K.reverse(y_pred, 3)

            vgg = VGG16(include_top=False, weights='imagenet', input_shape=(self.resize_size, self.resize_size, 3)) 
            loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output) 
            loss_model.trainable = False
            return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

        '''
        [u'block1_conv1', u'block1_conv2', u'block1_pool', 
        u'block2_conv1', u'block2_conv2', u'block2_pool', 
        u'block3_conv1', u'block3_conv2', u'block3_conv3', u'block3_pool', 
        u'block4_conv1', u'block4_conv2', u'block4_conv3', u'block4_pool', 
        u'block5_conv1', u'block5_conv2', u'block5_conv3', u'block5_pool']
        '''
        
        def perceptual_loss_multiple_layers(y_true, y_pred):
            #y_true = array_to_img(y_true)
            #y_pred = array_to_img(y_pred)
            #y_true = preprocess_input(y_true)
            #y_pred = preprocess_input(y_pred)
            vgg = VGG16(include_top=False, weights='imagenet', input_shape=(self.resize_size, self.resize_size, 3)) 
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

            loss1 = K.mean(K.abs(loss_model_1(y_true) - loss_model_1(y_pred)))
            loss2 = K.mean(K.abs(loss_model_2(y_true) - loss_model_2(y_pred)))
            loss3 = K.mean(K.abs(loss_model_3(y_true) - loss_model_3(y_pred)))
            loss4 = K.mean(K.abs(loss_model_4(y_true) - loss_model_4(y_pred)))
            loss5 = K.mean(K.abs(loss_model_5(y_true) - loss_model_5(y_pred)))

            loss = (1*loss1 + 2*loss2 + 3*loss3 + 3*loss4 + 3*loss5) / 12.0

            return loss
        
        def custom_loss(y_true, y_pred): 
            weight_p = 0.9
            weight_m = 0.1
            #y_true_p = preprocess_input(y_true)
            #y_pred_p = preprocess_input(y_pred)
            vgg = VGG16(include_top=False, weights='imagenet', input_shape=(self.resize_size, self.resize_size, 3)) 
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

            loss1 = K.mean(K.abs(loss_model_1(y_true) - loss_model_1(y_pred)))
            loss2 = K.mean(K.abs(loss_model_2(y_true) - loss_model_2(y_pred)))
            loss3 = K.mean(K.abs(loss_model_3(y_true) - loss_model_3(y_pred)))
            loss4 = K.mean(K.abs(loss_model_4(y_true) - loss_model_4(y_pred)))
            loss5 = K.mean(K.abs(loss_model_5(y_true) - loss_model_5(y_pred)))

            loss = weight_p * ((loss1 + loss2 + loss3 + loss4 + loss5) / 5.0) + weight_m * K.mean(K.square(y_pred - y_true), axis=-1)

            return loss

        def custom_loss_one_layer(y_true, y_pred): 
            weight_p = 0.9999
            weight_m = 0.0001
            #y_true_p = preprocess_input(y_true)
            #y_pred_p = preprocess_input(y_pred)
            vgg = VGG16(include_top=False, weights='imagenet', input_shape=(self.resize_size, self.resize_size, 3)) 
            
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
            vgg = VGG16(include_top=False, weights='imagenet', input_shape=(self.resize_size, self.resize_size, 3)) 
            
            loss_model_3 = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output) 
            loss_model_3.trainable = False

            loss_model_5 = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv3').output) 
            loss_model_5.trainable = False

            
            loss3 = K.mean(K.square(loss_model_3(y_true) - loss_model_3(y_pred)))
            loss5 = K.mean(K.square(loss_model_5(y_true) - loss_model_5(y_pred)))

            loss = weight_p * ((loss3 + loss5) / 2.0) + weight_m * K.mean(K.square(y_pred - y_true), axis=-1)

            return loss
        
        
        #def wasserstein_loss(y_true, y_pred):
        #    return K.mean(y_true * y_pred)

        #loss = [perceptual_loss, 'mse']
        #loss_weights = [100, 1]

        input_shirt = Input((self.resize_size, self.resize_size, 3))
        input_mask = Input((self.resize_size, self.resize_size, 1))
        inputs = concatenate([input_shirt, input_mask])

        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        print ("conv1 shape:",conv1.shape)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        print ("conv1 shape:",conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print ("pool1 shape:",pool1.shape)

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        print ("conv2 shape:",conv2.shape)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        print ("conv2 shape:",conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print ("pool2 shape:",pool2.shape)

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        print ("conv3 shape:",conv3.shape)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        print ("conv3 shape:",conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print ("pool3 shape:",pool3.shape)

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        #conv10 = Conv2D(3, 3, activation = 'sigmoid')(conv9)

        model = Model(inputs = [input_shirt, input_mask], output = conv9)

        model.compile(optimizer = Adam(lr = 1e-4), loss = custom_loss_one_layer, metrics = [custom_loss_one_layer, 'accuracy'])

        return model


    def train(self):

        
        def perceptual_loss(y_true, y_pred): 
            #y_true = preprocess_input(y_true)
            #y_pred = preprocess_input(y_pred)
            vgg = VGG16(include_top=False, weights='imagenet', input_shape=(self.resize_size, self.resize_size, 3)) 
            loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block4_conv3').output) 
            loss_model.trainable = False
            return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

        print("loading data")
        imgs_train, imgs_train_mask, imgs_train_label, imgs_test, imgs_test_mask = self.load_data()
        print("loading data done")

        if new_train:
            model = self.get_unet()
            print("got a new unet")
        else:
            list_of_models = glob.glob(pervious_model_dir + '/*.hdf5')
            if list_of_models:
                latest_model = max(list_of_models, key=os.path.getctime)
                model = load_model(latest_model, custom_objects={'custom_loss': custom_loss})
                print("train from model: " + latest_model)
            else:
                model = self.get_unet()
                print("got a new unet")
        filepath = os.path.join(model_dir, "unet-{epoch:02d}-{custom_loss_one_layer:.2f}-{val_acc:.2f}.hdf5")
        model_checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1, save_best_only=True)
        callbacks_list = [model_checkpoint]

        log_filepath = "./keras_log_perceptual"
        tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=1)
        cbks = [tb_cb]
        print('Fitting model...')
        model.fit([imgs_train, imgs_train_mask], imgs_train_label, batch_size=4, epochs=30, verbose=1, validation_split=0.2, shuffle=True, callbacks=callbacks_list)

        '''
        fig1 = pyplot.figure()
        pyplot.plot(history.history['loss'])
        fig1.savefig(os.path.join(train_log, 'loss_fig.png'))

        fig2 = pyplot.figure()
        pyplot.plot(history.history['accuracy'])
        fig2.savefig(os.path.join(train_log, 'val_acc_fig.png'))
        '''

        print('predict test data')
        imgs_test_result = model.predict([imgs_test, imgs_test_mask], batch_size=1, verbose=1)
        np.save( os.path.join(result_dir, 'imgs_test_result.npy'), imgs_test_result)
        np.save( os.path.join(result_dir, 'imgs_test.npy'), imgs_test)
        np.save( os.path.join(result_dir, 'imgs_test_mask.npy'), imgs_test_mask)

    def deprocess_img(img):
        mean = [103.939, 116.779, 123.68]
        img[..., 0] += mean[0]
        img[..., 1] += mean[1]
        img[..., 2] += mean[2]
        img = img[..., ::-1]
        return img

    def save_img(self):
        print("array to image")
        imgs = np.load(os.path.join(result_dir, 'imgs_test_result.npy'))
        for i in range(imgs.shape[0]):
            img = imgs[i]

            #img = deprocess_img(img)

            #img = array_to_img(img)
            save_dir = os.path.join(result_dir, 'test_results')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            #img.save( os.path.join( save_dir, "%d.jpg"%(i) ) )
            #img *= 255
            img = np.clip(img, 0, 255).astype('uint8')
            imsave(os.path.join( save_dir, "%d.jpg"%(i) ), img)

        imgs = np.load(result_dir + '/imgs_test.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]

            #img = deprocess_img(img)

            #img = array_to_img(img)
            save_dir = os.path.join(result_dir, 'test_images')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            #img.save( os.path.join( save_dir, "%d.jpg"%(i) ) )
            #img *= 255
            img = np.clip(img, 0, 255).astype('uint8')
            imsave(os.path.join( save_dir, "%d.jpg"%(i) ) , img)

        imgs = np.load(result_dir + '/imgs_test_mask.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            save_dir = os.path.join(result_dir, 'test_masks')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img.save( os.path.join( save_dir, "%d.jpg"%(i) ) )




if __name__ == '__main__':
    new_train = True
    pervious_model_dir = "./model_perceptualloss_256_conv4"
    model_dir = "./model_perceptualloss_256_conv5_conv5_mse"
    result_dir = "./results_perceptualloss_256_conv5_conv5_mse"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    myunet = myUnet()
    myunet.train()
    myunet.save_img()







