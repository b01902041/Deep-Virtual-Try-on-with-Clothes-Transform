from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import gc
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from scipy.misc import imsave
#import cv2
#from libtiff import TIFF

class dataProcess(object):
    

    def __init__(self, out_rows, out_cols, data_path = "./MVC_image_pairs_resize_new/shirts_1", mask_path = "./MVC_image_pairs_resize_new/fc8_mask_5_modified", label_path = "./MVC_image_pairs_resize_new/shirts_5", test_data_path = "./test/shirts_1", test_mask_path = "./test/fc8_mask_5_modified", npy_path = "./npydata", img_type = "jpg", mask_type = ""):

        """
        
        """

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.mask_path = mask_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_data_path = test_data_path
        self.test_mask_path = test_mask_path
        self.npy_path = npy_path
        self.resize_size = 256

    def create_train_data(self):
        i = 0
        
        print('-'*30)
        print('Creating training images...')
        print('-'*30)
        imgs = glob.glob(self.data_path+"/*."+self.img_type)
        print(len(imgs))
        
        imgdatas = np.ndarray((len(imgs), self.resize_size, self.resize_size, 3), dtype=np.float32)
        imgmasks = np.ndarray((len(imgs), self.resize_size, self.resize_size, 1), dtype=np.float32)
        imglabels = np.ndarray((len(imgs), self.resize_size, self.resize_size, 3), dtype=np.float32)

        if not os.path.exists(self.npy_path):
            os.mkdir(self.npy_path)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/")+1:]
            maskname = midname.split("_")[0] + "_5.jpg"

            img = load_img(self.data_path + "/" + midname, grayscale = False)
            img_mask = load_img(self.mask_path + "/" + maskname, grayscale = True)
            label = load_img(self.label_path + "/" + maskname, grayscale = False)

            img = img.resize( (self.resize_size, self.resize_size), Image.BILINEAR )
            img_mask = img_mask.resize( (self.resize_size, self.resize_size), Image.BILINEAR )
            label = label.resize( (self.resize_size, self.resize_size), Image.BILINEAR )

            #print('img is: ' )
            #print(img)


            img = img_to_array(img)
            img_mask = img_to_array(img_mask)
            label = img_to_array(label)

            #img /= 255
            #label /= 255
            #img = preprocess_input(img)
            #label = preprocess_input(label)

            #print('img_processed is: ' )
            #print(img)
            '''
            mean = [103.939, 116.779, 123.68]
            img[..., 0] += mean[0]
            img[..., 1] += mean[1]
            img[..., 2] += mean[2]
            img = img[..., ::-1]
            img = np.clip(img, 0, 255).astype('uint8')
            imsave('./gray.jpg', img)
            '''

            #img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            #label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            #img = np.array([img])
            #label = np.array([label])
            imgdatas[i] = img
            imgmasks[i] = img_mask
            imglabels[i] = label

            #print('imgdatas[i] is: ' )
            #print(imgdatas[i])
            
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            '''
            if num_of_imgs == num_imgs_a_pack:
                if not (os.path.isfile(self.npy_path + '/imgs_train_' + str(i) + '.npy')):
                    np.save(self.npy_path + '/imgs_train_' + str(i) + '.npy', imgdatas)
                    np.save(self.npy_path + '/imgs_train_mask_' + str(i) + '.npy', imgmasks)
                    np.save(self.npy_path + '/imgs_train_label_'  + str(i) + '.npy', imglabels)
                    print('Saving to %d .npy files done.', i)
                #imgdatas = np.ndarray((num_imgs_a_pack, self.out_rows,self.out_cols, 3), dtype=np.uint8)
                #imgmasks = np.ndarray((num_imgs_a_pack, self.out_rows,self.out_cols, 1), dtype=np.uint8)
                #imglabels = np.ndarray((num_imgs_a_pack, self.out_rows,self.out_cols, 3), dtype=np.uint8)
                num_of_imgs = 0
                gc.collect()
            '''
            i += 1
        '''
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_train_mask.npy', imgmasks)
        np.save(self.npy_path + '/imgs_train_label.npy', imglabels)
        print('Saving to %d .npy files done.', i)
        '''
        return imgdatas, imgmasks, imglabels


    def create_test_data(self):
        i = 0

        print('-'*30)
        print('Creating test images...')
        print('-'*30)
        imgs = glob.glob(self.test_data_path+"/*."+self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.resize_size, self.resize_size, 3), dtype=np.float32)
        imgmasks = np.ndarray((len(imgs), self.resize_size, self.resize_size, 1), dtype=np.float32)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/")+1:]
            maskname = midname.split("_")[0] + "_5.jpg"
            img = load_img(self.test_data_path + "/" + midname, grayscale = False)
            img = img.resize( (self.resize_size, self.resize_size), Image.BILINEAR )
            img = img_to_array(img)
            #img /= 255
            #img = preprocess_input(img)

            img_mask = load_img(self.test_mask_path + "/" + maskname, grayscale = True)
            img_mask = img_mask.resize( (self.resize_size, self.resize_size), Image.BILINEAR )
            img_mask = img_to_array(img_mask)


            #img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            #img = np.array([img])
            imgdatas[i] = img
            imgmasks[i] = img_mask
            i += 1
        print('loading done')
        '''
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        np.save(self.npy_path + '/imgs_test_mask.npy', imgmasks)
        print('Saving to test .npy files done.')
        '''
        return imgdatas, imgmasks

if __name__ == "__main__":

    #aug = myAugmentation()
    #aug.Augmentation()
    #aug.splitMerge()
    #aug.splitTransform()
    mydata = dataProcess(641, 641)
    mydata.create_train_data()
    mydata.create_test_data()
    #imgs_train,imgs_mask_train = mydata.load_train_data()
    #print imgs_train.shape,imgs_mask_train.shape