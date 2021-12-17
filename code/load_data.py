import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_training(data_path="../training_data"):

    song_dir = os.listdir(data_path)

    input_texts = []
    target_texts = []
    target_tokens = []

    for song in song_dir:
        measures = os.listdir("{}/{}".format(data_path,song))
        for i in range(1,len(measures)-1):
            #gathering of input texts
            input_file = open("{}/{}/{}.measure_{}.txt".format(data_path,song,song,i))
            input_text = list(input_file)
            input_texts.append(input_text)

            #gathering of target texts
            target_file = open("{}/{}/{}.measure_{}.txt".format(data_path,song,song,i+1))
            target_text = list(target_file)
            target_texts.append(target_text)
            
            for token in input_text:
                if token not in target_tokens:
                    target_tokens.append(token)
        
        #we add the potentially new characters present in every last measure, 
        #as we didn't go through them in the previous loop.
        for token in target_text:
            if token not in target_tokens:
                target_tokens.append(token)

        target_tokens = sorted(target_tokens)

        tokens_file = open("../tokens/tokens_list.txt","w")
        for token in target_tokens:
            tokens_file.write(token)
        tokens_file.close()

    
    return input_texts, target_texts, target_tokens

def load_test(data_path="../test_data"):

    song_dir = os.listdir(data_path)
    input_texts = []
    target_texts = []
    headers = []

    for song in song_dir:
        measures = os.listdir("{}/{}".format(data_path,song))
        header = open("{}/{}/{}.header.txt".format(data_path,song,song))
        for i in range(1,len(measures)-1):
            #gathering of input texts
            input_file = open("{}/{}/{}.measure_{}.txt".format(data_path,song,song,i))
            input_text = list(input_file)
            input_texts.append(input_text)

            #gathering of target texts
            target_file = open("{}/{}/{}.measure_{}.txt".format(data_path,song,song,i+1))
            target_text = list(target_file)
            target_texts.append(target_text)
            headers.append(header)


    return input_texts, target_texts, headers