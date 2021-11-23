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
    idx_token_input = 0
    idx_token_target = 0
    token_input=""
    token_target=""
    for char in input_text:
        token_input+=char
        if token_input in target_tokens:
            encoder_input_data[i, idx_token_input, token_index[token_input]] = 1.0
            token_input=""
            idx_token_input+=1
    for char in target_text:
    # decoder_target_data is ahead of decoder_input_data by one timestep
        token_target+=char
        if token_target in target_tokens:
            decoder_input_data[i, idx_token_target, token_index[token_target]] = 1.0
            token_target=""
            idx_token_input+=1
        decoder_input_data[i, idx_token_target, token_index[char]] = 1.0
        if idx_token_target>0:
            decoder_target_data[i, idx_token_target-1, token_index[char]] = 1.0

