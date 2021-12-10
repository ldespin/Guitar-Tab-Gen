import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_path="../training_data", sample_size=70, rand_state=None):

    rng=np.random.default_rng(rand_state)
    
    song_dir = os.listdir(data_path)
    if sample_size>len(song_dir):
        sample_size=len(song_dir)

    input_texts = []
    target_texts = []
    target_tokens = []

    song_dir_sample = rng.choice(song_dir, sample_size)

    for song in song_dir_sample:
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