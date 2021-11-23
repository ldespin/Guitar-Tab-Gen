import numpy as np
import os
import re
import matplotlib.pyplot as plt
from get_tokens import target_tokens
from get_tokens import input_texts
from get_tokens import target_texts

def count_tokens(text, target):
    count = 0
    token = ''
    for char in text:
        token+=char
        if token in target:
            count+=1
            token = ''
    return count

def max_tokens(texts, target):
    return np.max([count_tokens(text, target) for text in texts])

num_tokens = len(target_tokens)
max_encoder_seq_length = max_tokens(input_texts, target_tokens)
token_index = dict([(char, i) for i, char in enumerate(target_tokens)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t,token_input in enumerate(input_text.split('\n')):
        encoder_input_data[i, t, token_index[token_input]] = 1.0
    for t,token_target in enumerate(target_text.split('\n')):
    # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, token_index[token_target]] = 1.0
        if t>0:
            decoder_target_data[i, t-1, token_index[token_target]] = 1.0

