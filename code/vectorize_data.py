import numpy as np
import os

def vectorize_training(input_text, target_text):
    target_tokens = list(open("../tokens/tokens_list.txt"))
    max_tokens = list(open("../tokens/max_tokens.txt"))[0]
    num_tokens = len(target_tokens)
    token_index = dict([(token, i) for i, token in enumerate(target_tokens)])

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

    return encoder_input_data, decoder_input_data, decoder_target_data


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
