from keras.models import Model
from keras import Input, layers
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
import numpy as np
from keras.layers.merge import add


class Image_Captioning_Model:
    model = None
    def __init__(self,glove_path,max_length,vocab_size,wordtoix):
        # load glove vectors
        embeddings_index = self.get_embeddings_index(glove_path)
        # get the embeddings dimension
        embedding_dim,embedding_matrix = self.get_embeddings_dimension(vocab_size,embeddings_index,wordtoix)
        self.model = self.build_model(max_length,vocab_size,embedding_dim,embedding_matrix)
        self.compile_model(self.model)

    #to build the model
    def build_model(self,max_length,vocab_size,embedding_dim,embedding_matrix):
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        model.layers[2].set_weights([embedding_matrix])
        model.layers[2].trainable = False
        return model

    #to compile the model
    def compile_model(self,model):
        model.compile(loss='categorical_crossentropy', optimizer='adam')

    #load the glove vectors file and get the index
    def get_embeddings_index(self,glove_path):
        embeddings_index = {}
        f = open(glove_path, encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asanyarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    #get the embeddings dimension
    def get_embeddings_dimension(self,vocab_size,embeddings_index,wordtoix):
        embedding_dim = 200
        #get 200-dimension dense vector for each of the 10000 words in out vocabulary
        embedding_matrix = np.zeros((vocab_size,embedding_dim))
        for word,i in wordtoix.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                #words not found in the embedding index will be all zeros
                embedding_matrix[i] = embedding_vector
        return embedding_dim,embedding_matrix