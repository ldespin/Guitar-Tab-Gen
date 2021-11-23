import numpy as np
import os
import re
import matplotlib.pyplot as plt

input_texts = []
target_texts = []
target_tokens = ['\n']

data_path = "./sample"
song_dir = os.listdir(data_path)


#input_texts will contain all measures but the last one, for every song, while the target_texts will be composed 
#of all measures expect the first one. In this way, target text represents the measure following the input one.
for song in song_dir:
    measures = os.listdir("{}/{}".format(data_path,song))
    for i in range(1,len(measures)-1):
        #gathering of input texts
        input_file = open("{}/{}/{}.measure_{}.txt".format(data_path,song,song,i))
        input_text = input_file.read()
        tokens = input_file.split('\n')
        input_texts.append(input_text)

        #gathering of target texts
        target_file = open("{}/{}/{}.measure_{}.txt".format(data_path,song,song,i+1))
        target_text = target_file.read()
        target_texts.append(target_text)
        
        for token in tokens:
            if token not in target_tokens:
                target_tokens.append(token)
        
    #we add the potentially new characters present in every last measure, 
    #as we didn't go through them in the previous loop.
    for token in target_text.split('\n'):
        if token not in target_tokens:
            target_tokens.append(token)

    target_tokens = sorted(target_tokens)