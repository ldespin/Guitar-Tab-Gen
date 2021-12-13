import numpy as np

def encode_data(input_texts, target_tokens, max_encoder_seq_length):

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
