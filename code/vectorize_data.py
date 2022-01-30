import numpy as np
import os

def vectorize_training(target_tokens, max_tokens,data_path_raw="../training_data/raw", data_path_vec="../training_data/vectorized"):
    
    song_dir = os.listdir(data_path_raw)

    token_index = dict([(token, i) for i, token in enumerate(target_tokens)])
    num_tokens = len(target_tokens)

    idx = 0

    for song in song_dir:
        measures = os.listdir("{}/{}".format(data_path_raw,song))
        for j in range(1,len(measures)-1):
            input_file = open("{}/{}/{}.measure_{}.txt".format(data_path_raw,song,song,j))
            input_text = list(input_file)

            target_file = open("{}/{}/{}.measure_{}.txt".format(data_path_raw,song,song,j+1))
            target_text = list(target_file)

            encoder_input_data = np.zeros(
                (max_tokens, num_tokens), dtype="float32"
            )
            decoder_input_data = np.zeros(
                (max_tokens, num_tokens), dtype="float32"
            )
            decoder_target_data = np.zeros(
                (max_tokens, num_tokens), dtype="float32"
            )

            for t,token_input in enumerate(input_text):
                if token_input in target_tokens:
                    encoder_input_data[t, token_index[token_input]] = 1.0
            for t,token_target in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
                if token_target in target_tokens:
                    decoder_input_data[t, token_index[token_target]] = 1.0
                    if t>0:
                        decoder_target_data[t-1, token_index[token_target]] = 1.0
            
            with open(f'{data_path_vec}/encoder_input_{idx}.npy', 'wb') as f:
                np.save(f,encoder_input_data)

            with open(f'{data_path_vec}/decoder_input_{idx}.npy', 'wb') as f:
                np.save(f,decoder_input_data)
            
            with open(f'{data_path_vec}/decoder_target_{idx}.npy', 'wb') as f:
                np.save(f,decoder_target_data)


def vectorize_test(input_texts, target_tokens, max_encoder_seq_length):

    num_tokens = len(target_tokens)
    token_index = dict([(token, i) for i, token in enumerate(target_tokens)])
    encoded_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_tokens), dtype="float32"
    )


    for i, input_text in enumerate(input_texts):
        for t,token_input in enumerate(input_text):
            if token_input in token_index:
                encoded_data[i, t, token_index[token_input]] = 1.0

    return encoded_data


def unvectorize_test(encoded_data, target_tokens):

    token_index = dict([(i, token) for i, token in enumerate(target_tokens)])
    decoded_data = []

    measures, lines, tokens = encoded_data.shape

    for measure in range(measures):
        decoded_measure = []
        for line in range(lines):
            for token in range(tokens):
                if encoded_data[measure, line, token]==1.0:
                    print(token_index[token])
                    decoded_measure.append(token_index[token])
        decoded_data.append(decoded_measure)
    return decoded_data
