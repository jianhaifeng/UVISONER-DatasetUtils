from utils.ImageCaptioning import Dataset

token_path = 'D:/dataset/image_caption/Flickr8k_text/Flickr8k.token.txt'
description_path = 'D:/dataset/image_caption/descriptions.txt'
train_dataset_path = 'D:/dataset/image_caption/Flickr8k_text/Flickr_8k.trainImages.txt'
test_dataset_path = 'D:/dataset/image_caption/Flickr8k_text/Flickr_8k.testImages.txt'
images_directory = 'D:/dataset/image_caption/Flickr8k_Dataset/'
image_suffix = 'jpg'

dataset = Dataset(token_path,
                 description_path,
                 train_dataset_path,
                 test_dataset_path,
                 images_directory,
                 image_suffix)






