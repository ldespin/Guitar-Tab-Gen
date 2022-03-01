from load_data import get_tokens
from load_data import DataGenerator
import lstm_model
import os 

if __name__=="__main__":

    instr = "distorted0"

    target_tokens, max_tokens = get_tokens(instr=instr)

    num_tokens = len(target_tokens)

    measures = os.listdir("../training_data/raw")

    n=999
    training_ids = []
    for i in range(999):
        instr_present = False
        song = []
        for file in measures:
            if f"_{i}_" in file:
                song.append(file)
                f = open("file","r")
                for token in f.list():
                    if instr in token:
                        instr_present = True
        if instr_present:
            for j in range(len(song)-1):
                training_ids.append([i,j])
        
    training_generator = DataGenerator(training_ids)

    lstm_model.train_lstm(training_generator, num_tokens)

