from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from pickle import dump, load
import numpy as np
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

class InceptionV3_Encode:

    encoding_train = None
    encoding_test = None
    train_features = None


    def __init__(self,images_directory,train_features_path,test_features_path,train_images,test_images):
        # load the inception v3 model
        model = InceptionV3(weights='imagenet')
        # create a new model,by removing the last layer (output layer) from the inception v3
        model_new = Model(model.input,model.layers[-2].output)
        # call the function to encode all the train images
        self.encoding_train = self.encode_images_inceptionv3(images_directory,train_images,model_new,train_features_path)
        # call the function to encode all the test images
        self.encoding_test = self.encode_images_inceptionv3(images_directory,test_images,model_new,test_features_path)
        self.train_features = load(open(train_features_path,'rb'))

    #encode the images by Model
    def encode_images_inceptionv3(self,images_directory,images,model_new,features_path):
        encoding_temp = {}
        for image_1 in images:
            encoding_temp[image_1[len(images_directory):]] = self.encode(image_1, model_new)
        # save the bottleneck train features to disk
        with open(features_path, 'wb') as encoded_pickle:
            dump(encoding_temp, encoded_pickle)
        return encoding_temp

    #function to encode a given image into vector of size(2048,)
    def encode(self,image_1,model_new):
        image_2 = self.inception_v3_preprocess(image_1)#preprocess the image
        fea_vec = model_new.predict(image_2)#get the encoding vector for the image
        fea_vec = np.reshape(fea_vec,fea_vec.shape[1])#reshape from (1, 2048) to (2048, )
        return fea_vec

    def inception_v3_preprocess(self,image_path):
        #convert all the images to size 299*299 as expected by the inception v3 model
        image_3 = image.load_img(image_path,target_size=(299,299))
        #convert PIL image to numpy array of 3-dimensions
        x = image.img_to_array(image_3)
        #add one more dimension
        x = np.expand_dims(x,axis=0)
        #preprocess the images using preprocess_input() from inception module
        x = preprocess_input(x)
        return x