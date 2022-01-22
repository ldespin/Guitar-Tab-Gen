import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_training(data_path="../training_data",guitar_only="on"):

    song_dir = os.listdir(data_path)

    input_texts = []
    target_texts = []
    input_texts_guitar = []
    target_texts_guitar = []
    target_tokens = []

    for song in song_dir:
        measures = os.listdir("{}/{}".format(data_path,song))
        for i in range(1,len(measures)-1):
            #gathering of input texts
            input_file = open("{}/{}/{}.measure_{}.txt".format(data_path,song,song,i))
            input_text = list(input_file)
            input_text_guitar = []
            input_texts.append(input_text)

            #gathering of target texts
            target_file = open("{}/{}/{}.measure_{}.txt".format(data_path,song,song,i+1))
            target_text = list(target_file)
            target_text_guitar = []
            target_texts.append(target_text)
            
            for token in input_text:
                if guitar_only=="on" and "bass" not in token and "drums" not in token and "nfx" not in token:
                    if token not in target_tokens:
                        target_tokens.append(token)
                    input_text_guitar.append(token)

                elif guitar_only=="off":
                    if token not in target_tokens:
                        target_tokens.append(token)
            
            for token in target_text:
                if guitar_only=="on" and "bass" not in token and "drums" not in token and "nfx" not in token:
                    if token not in target_tokens:
                        target_tokens.append(token)
                    target_text_guitar.append(token)

                elif guitar_only=="off":
                    if token not in target_tokens:
                        target_tokens.append(token)
            
            input_texts_guitar.append(input_text_guitar)
            target_texts_guitar.append(target_text_guitar)

        target_tokens = sorted(target_tokens)

        tokens_file = open("../tokens/tokens_list.txt","w")
        for token in target_tokens:
            tokens_file.write(token)
        tokens_file.close()

    if guitar_only=="on":
        return input_texts_guitar, target_texts_guitar, target_tokens
    return input_texts, target_texts, target_tokens

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