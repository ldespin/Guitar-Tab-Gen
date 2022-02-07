from load_data import get_tokens
from load_data import DataGenerator
import lstm_model
import os 

if __name__=="__main__":

    target_tokens, max_tokens = get_tokens()

    num_tokens = len(target_tokens)

    measures = os.listdir("../training_data/raw")

    n=999
    training_ids = []
    for i in range(999):
        song = []
        for file in measures:
            if f"_{i}_" in file:
                song.append(file)
        for j in range(len(song)-1):
            training_ids.append([i,j])
        
    training_generator = DataGenerator(training_ids)

    

    #lstm_model.train_lstm(encoder_input_data, decoder_input_data, decoder_target_data, num_tokens)

