import string
import glob
from numpy import array

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class Dataset:

    images = None #all images
    train_images = None #the images of training
    test_images = None #the images of testing
    train_descriptions = None #the description of train
    wordtoix = None
    max_length = None
    vocab_size = None
    ixtoword = None

    def __init__(self,token_path,
                 description_path,#to save the description result
                 train_dataset_path,
                 test_dataset_path,
                 images_directory,
                 image_suffix):
        print('token_path:',token_path)
        print('description_path:', description_path)
        print('train_dataset_path:', train_dataset_path)
        print('test_dataset_path:', test_dataset_path)
        print('images_directory:', images_directory)
        print('image_suffix:', image_suffix)
        #load descriptions
        token_text = self.load_file(token_path)
        print(token_text[:300])
        #parse descriptions
        description_mappinglist = self.load_descriptions(token_text)
        print('Loaded: %d ' % len(description_mappinglist))
        #clean descriptions
        self.clean_descriptions(description_mappinglist)
        #summarize vocabulary
        vocabulary = self.to_vocabulary(description_mappinglist)
        print('Original Vocabulary Size: %d' % len(vocabulary))
        #save the description
        self.save_descriptions(description_mappinglist,description_path)
        #load the train dataset
        train = self.load_set(train_dataset_path)
        print('Dataset: %d' % len(train))
        #load all images
        images = glob.glob(images_directory+'*.'+image_suffix)
        #load train image names
        train_imagenames = set(open(train_dataset_path,'r').read().strip().split('\n'))
        print('train_imagenames:',len(train_imagenames))
        #load train images
        self.train_images = self.get_images(images_directory,images,train_imagenames)
        print('self.train_images:',len(self.train_images))
        #load test images names
        test_imagenames = set(open(test_dataset_path,'r').read().strip().split('\n'))
        print('test_imagenames:', len(test_imagenames))
        self.test_images = self.get_images(images_directory,images,test_imagenames)
        print('self.test_images:', len(self.test_images))
        self.train_descriptions = self.load_clean_descriptions(description_path,train)
        print('Descriptions: train=%d' % len(self.train_descriptions))
        #create a list of all the training captions
        all_train_captions = self.get_train_captions(self.train_descriptions)
        print('all_train_captions: train=%d' % len(all_train_captions))
        #consider only words which occur at least 10 times in the corpus
        word_counts, nsents, vocab, self.ixtoword, self.wordtoix, ix, self.vocab_size = self.define_vocabulary_config(all_train_captions)

        #determine the maximum sequence length
        self.max_length = self.max_length(self.train_descriptions)
        print('Description Length: %d' % self.max_length)


    #get the data generator
    def get_train_data_generator(self,num_images_per_batch,train_features):
        return self.data_generator(self.train_descriptions,
                                   train_features,self.wordtoix,self.max_length,num_images_per_batch,self.vocab_size)

    #consider only words which occur at least 10 times in the corpus
    def define_vocabulary_config(self,all_train_captions):
        word_count_threshold = 10
        word_counts = {}
        nsents = 0
        for sent in all_train_captions:
            nsents += 1
            for w in sent.split(' '):
                word_counts[w] = word_counts.get(w, 0) + 1
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))
        ixtoword = {}
        wordtoix = {}
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        vocab_size = len(ixtoword) + 1
        return word_counts,nsents,vocab,ixtoword,wordtoix,ix,vocab_size

    #create a list of all the training captions
    def get_train_captions(self,train_descriptions):
        all_train_captions = []
        for key, val in train_descriptions.items():
            for cap in val:
                all_train_captions.append(cap)
        return all_train_captions

    #get the images
    def get_images(self,images_directory,images,image_names):
        temp_images = []
        for i in images:
            if i[len(images_directory):] in image_names:
                temp_images.append(i)
        return temp_images

    def load_file(self,path):
        try:
            file = open(path,'r')
            text = file.read()
            file.close()
        except IOError:
            print('An exception occurred while reading the file at %s !',path)
        return text

    def load_descriptions(self,token_text):
        mapping = dict()
        for line in token_text.split('\n'):
            #split line by white space
            tokens = line.split()
            if len(line) < 2:
                continue
            #get the image id and the description
            image_id,image_description = tokens[0],tokens[1:]
            #get the filename from image_id
            image_id = image_id.split('.')[0]
            #convert description tokens back to string
            image_description = ' '.join(image_description)
            #create the list if needed
            if image_id not in mapping:
                mapping[image_id] = list()
            #store the description to mapping list by image_id
            mapping[image_id].append(image_description)
        return mapping

    def clean_descriptions(self,description_mappinglist):
        #prepare translation table for removing punctuation
        table = str.maketrans('','',string.punctuation)
        for key,desc_list in description_mappinglist.items():
            for i in range(len(desc_list)):
                desc = desc_list[i]
                #tokenize
                desc = desc.split()
                #convert to lower case
                desc = [word.lower() for word in desc]
                #remove punctuation from each token
                desc = [w.translate(table) for w in desc]
                #remove hanging 's' and 'a'
                desc = [word for word in desc if len(word) > 1]
                #remove tokens with numbers in them
                desc = [word for word in desc if word.isalpha()]
                #store as string
                desc_list[i] = ' '.join(desc)

    #convert the loaded descriptions into a vocabulary of words
    def to_vocabulary(self,description_mappinglist):
        #build a list of all description strings
        all_desc = set()
        for key in description_mappinglist.keys():
            [all_desc.update(d.split()) for d in description_mappinglist[key]]
        return all_desc

    #save descriptions to file,one per line
    def save_descriptions(self,description_mappinglist,description_filepath):
        lines = list()
        for key,desc_list in description_mappinglist.items():
            for desc in desc_list:
                lines.append(key+' '+desc)
            data = '\n'.join(lines)
            file = open(description_filepath,'w')
            file.write(data)
            file.close()

    #load a pre-defined list of photo identifiers
    def load_set(self,filepath):
        doc = self.load_file(filepath)
        dataset = list()
        #process line by line
        for line in doc.split('\n'):
            #skip empty lines
            if len(line) < 1:
                continue
            #get the image identifier
            identifier = line.split('.')[0]
            dataset.append(identifier)
        return set(dataset)

    #load clean descriptions into memory
    def load_clean_descriptions(self,description_filepath,dataset):
        #load document
        doc = self.load_file(description_filepath)
        descriptions = dict()
        for line in doc.split('\n'):
            #split line by white space
            tokens = line.split()
            #split id from description
            image_id,image_desc = tokens[0],tokens[1:]
            #skip images not in the set
            if image_id in dataset:
                #create list
                if image_id not in descriptions:
                    descriptions[image_id] = list()
                #wrap description in tokens
                desc = 'startseq '+' '.join(image_desc)+' endseq'
                #store
                descriptions[image_id].append(desc)
        return descriptions

    #convert a dictionary of clean descriptions to a list of descriptions
    def to_lines(self,descriptions):
        all_desc = list()
        for key in descriptions.keys():
            [all_desc.append(d) for d in descriptions[key]]
        return all_desc

    #calculate the length of the description with the most words
    def max_length(self,descriptions):
        lines = self.to_lines(descriptions)
        return max(len(d.split()) for d in lines)

    #data generator,intended to be used in a call to model.fit_generator()
    def data_generator(self,descriptions,images,wordtoix,max_length,num_images_per_batch,vocab_size):
        X1,X2,y = list(),list(),list()
        n = 0
        #loop for ever over images
        while 1:
            for key,desc_list in descriptions.items():
                n += 1
                #retrieve the photo feature
                image = images[key+'.jpg']
                for desc in desc_list:
                    #encode the sequence
                    seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                    #split one sequence into multiple X,y pairs
                    for i in range(1,len(seq)):
                        #split into input and output pair
                        in_seq,out_seq = seq[:i],seq[i]
                        #pad input sequence
                        in_seq = pad_sequences([in_seq],maxlen=max_length)[0]
                        #encode output sequence
                        out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
                        #store
                        X1.append(image)
                        X2.append(in_seq)
                        y.append(out_seq)
                #yield the batch data
                if n == num_images_per_batch:
                    yield [[array(X1),array(X2)],array(y)]
                    X1,X2,y = list(),list(),list()
                    n = 0