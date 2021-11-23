import numpy as np
import os
import re
import matplotlib.pyplot as plt

input_texts = []
target_tokens = ['\n']

data_path = "./sample"
song_dir = os.listdir(data_path)

#input_texts will contain all measures but the last one, for every song, while the target_texts will be composed 
#of all measures expect the first one. In this way, target text represents the measure following the input one.
for song in song_dir:
    measures = os.listdir("{}/{}".format(data_path,song))
    for i in range(1,len(measures)):
        #gathering of input texts
        input_file = open("{}/{}/{}.measure_{}.txt".format(data_path,song,song,i))
        input_text = input_file.read()
        tokens = input_text.split("\n")
        input_texts.append(input_text)
        
        #we list all the characters that exist in our dataset
        for token in input_text:
            if token not in target_tokens:
                target_tokens.append(token)
        
        for token in tokens:
            if token not in target_tokens:
                target_tokens.append(token)
                if not is_number(token):
                    target_tokens_dig.append(token)
                    

X = ["Characters","Tokens","Tokens + Digits"]
Y = [len(target_characters), len(target_tokens), len(target_tokens_dig)]

plt.bar(X,Y)
plt.xlabel("Encoding format of the input")
plt.ylabel("Number of tokens")
plt.show()

