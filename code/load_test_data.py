import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_path="../test_data"):

    song_dir = os.listdir(data_path)
    input_texts = []
    target_texts = []

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

    return input_texts, target_texts