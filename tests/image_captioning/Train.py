from utils.ImageCaptioning import Dataset
from tests.image_captioning.InceptionV3_Encode import InceptionV3_Encode
from tests.image_captioning.Image_Captioning_Model import Image_Captioning_Model

token_path = 'D:/dataset/image_caption/Flickr8k_text/Flickr8k.token.txt'
description_path = 'D:/dataset/image_caption/descriptions.txt'
train_dataset_path = 'D:/dataset/image_caption/Flickr8k_text/Flickr_8k.trainImages.txt'
test_dataset_path = 'D:/dataset/image_caption/Flickr8k_text/Flickr_8k.testImages.txt'
images_directory = 'D:/dataset/image_caption/Flickr8k_Dataset/'
image_suffix = 'jpg'

#init the dataset
dataset = Dataset(token_path,
                 description_path,
                 train_dataset_path,
                 test_dataset_path,
                 images_directory,
                 image_suffix)

glove_path = 'D:/dataset/image_caption/glove.6B.200d.txt'
#build the model
imageCaptioningModel = Image_Captioning_Model(glove_path,dataset.max_length,dataset.vocab_size,dataset.wordtoix)
model = imageCaptioningModel.model

train_features_path = 'D:/dataset/image_caption/Pickle/encoded_train_images.pkl'
test_features_path = 'D:/dataset/image_caption/Pickle/encoded_test_images.pkl'
#encode dataset
inceptionV3Encode = InceptionV3_Encode(images_directory,train_features_path,test_features_path,dataset.train_images,dataset.test_images)

#the model saved directory
model_path = 'D:/AI(CV)/sourcecode/0818/Image_caption/model_weights/'
#start training
epochs = 10
number_pics_per_bath = 3
steps = len(dataset.train_descriptions)//number_pics_per_bath
for i in range(epochs):
    generator = dataset.get_train_data_generator(number_pics_per_bath,inceptionV3Encode.train_features)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save(model_path+'model_' + str(i) + '.h5')

for i in range(epochs):
    generator = dataset.get_train_data_generator(number_pics_per_bath,inceptionV3Encode.train_features)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save(model_path+'model_' + str(i) + '.h5')

epochs = 10
number_pics_per_bath = 6
steps = len(dataset.train_descriptions)//number_pics_per_bath
for i in range(epochs):
    generator = dataset.get_train_data_generator(number_pics_per_bath,inceptionV3Encode.train_features)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    #model.save('./model_weights/model_' + str(i) + '.h5')

model.save_weights(model_path+'model_30.h5')
