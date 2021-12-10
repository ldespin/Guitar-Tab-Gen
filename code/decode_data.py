import numpy as np

def decode_data(encoded_data, target_tokens):

    num_tokens = len(target_tokens)
    token_index = dict([(token, i) for i, token in enumerate(target_tokens)])
    decoded_data = []

    measures, lines, tokens = encoded_data.shape

    for measure in range(measures):
        decoded_measure = []
        for line in range(lines):
            for token in range(tokens):
                if encoded_data[measure, line, token]==1.0:
                    decoded_measure.append(token_index[token])
        decoded_data.append(decoded_measure)
    return decoded_data
