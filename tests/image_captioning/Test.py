from utils.ImageCaptioning import Dataset
from tests.image_captioning.Image_Captioning_Model import Image_Captioning_Model
from pickle import dump, load
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import random

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
imageCaptioningModel = Image_Captioning_Model(glove_path,dataset.max_length,dataset.vocab_size,dataset.wordtoix)
model = imageCaptioningModel.model

model_path = 'D:/AI(CV)/sourcecode/0818/Image_caption/model_weights/'
model.load_weights(model_path+'model_30.h5')

test_features_path = 'D:/dataset/image_caption/Pickle/encoded_test_images.pkl'
with open(test_features_path, "rb") as encoded_pickle:
    encoding_test = load(encoded_pickle)

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(dataset.max_length):
        sequence = [dataset.wordtoix[w] for w in in_text.split() if w in dataset.wordtoix]
        sequence = pad_sequences([sequence], maxlen=dataset.max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = dataset.ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

z= random.randint(1,len(encoding_test.keys()))
pic = list(encoding_test.keys())[z]
image = encoding_test[pic].reshape((1,2048))
x=plt.imread(images_directory+pic)
plt.text(0,0,greedySearch(image))
plt.imshow(x)
plt.show()
