import os
import numpy as np
import keras
from tensorflow.keras.utils import Sequence
import vectorize_data

def get_tokens(data_path="../training_data/raw",guitar_only="on"):
    measures = os.listdir(data_path)

    target_tokens = []
    max_tokens = 0

    for measure in measures:
        #gathering of input texts
        input_file = open(f"{data_path}/{measure}")
        input_text = list(input_file)
        input_text_guitar = []
    
        for token in input_text:
                if guitar_only=="on" and "bass" not in token and "drums" not in token and "nfx" not in token:
                    input_text_guitar.append(token)
                    if token not in target_tokens:
                        target_tokens.append(token)
                elif guitar_only=="off":
                    if token not in target_tokens:
                        target_tokens.append(token)
        if len(input_text_guitar)>max_tokens:
            max_tokens = len(input_text_guitar)

    target_tokens = sorted(target_tokens)
    
    tokens_file = open("../tokens/tokens_list.txt","w")
    for token in target_tokens:
        tokens_file.write(token)

    max_tokens_file = open("../tokens/max_tokens.txt","w")
    max_tokens_file.write(str(max_tokens))

    return target_tokens, max_tokens


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, n_channels=1, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Generate data
        for i, ID in enumerate(list_IDs_temp[:-1]):
            # Store sample
            input_file = open(f'..training_data/raw/data_{ID[0]}_{ID[1]}.txt')
            input_text = list(input_file)

            target_file = open(f'..training_data/raw/data_{ID[0]}_{ID[1]+1}.txt')
            target_text = list(target_file)

            encoder_input_data, decoder_input_data, decoder_target_data = vectorize_data.vectorize_training(input_text, target_text)

        return [encoder_input_data, decoder_input_data], decoder_target_data

def load_test(data_path="../test_data",guitar_only="on"):

    song_dir = os.listdir(data_path)
    input_texts = []
    target_texts = []
    input_texts_guitar = []
    target_texts_guitar = []
    headers = []

    for song in song_dir:
        measures = os.listdir("{}/{}".format(data_path,song))
        header = open("{}/{}/{}.header.txt".format(data_path,song,song))
        for i in range(1,len(measures)-1):
            #gathering of input texts
            
            input_file = open("{}/{}/{}.measure_{}.txt".format(data_path,song,song,i))
            input_text = list(input_file)   
            input_texts.append(input_text)
            input_text_guitar = []
            for token in input_text:
                if guitar_only=="on" and "bass" not in token and "drums" not in token and "nfx" not in token:
                    input_text_guitar.append(token)

            #gathering of target texts
            target_file = open("{}/{}/{}.measure_{}.txt".format(data_path,song,song,i+1))
            target_text = list(target_file)
            target_texts.append(target_text)

            target_text_guitar = []
            for token in target_text:
                if guitar_only=="on" and "bass" not in token and "drums" not in token and "nfx" not in token:
                    target_text_guitar.append(token)

            headers.append(header)

        input_texts_guitar.append(input_text_guitar)
        target_texts_guitar.append(target_text_guitar)

    if guitar_only=="on":
        return input_texts_guitar, target_texts_guitar, headers
    return input_texts, target_texts, headers