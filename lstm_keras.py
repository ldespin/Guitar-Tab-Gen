import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

batch_size = 64  # Batch size for training.
epochs = 30  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = "./sample"

# Preparation of the data
input_texts = []
target_texts = []
target_characters = set()

song_dir = os.listdir(data_path)

#input_texts will contain all measures but the last one, for every song, while the target_texts will be composed 
#of all measures expect the first one. In this way, target text represents the measure following the input one.
for song in song_dir:
    measures = os.listdir("{}/{}".format(data_path,song))
    for i in range(1,len(measures)-1):
        #gathering of input texts
        input_file = open("{}/{}/{}.measure_{}.txt".format(data_path,song,song,i))
        input_text = input_file.read()
        input_texts.append(input_text)
        
        #gathering of target texts
        target_file = open("{}/{}/{}.measure_{}.txt".format(data_path,song,song,i+1))
        target_text = target_file.read()
        target_texts.append(target_text)
        
        #we list all the characters that exist in our dataset
        for char in input_text:
            if char not in target_characters:
                target_characters.add(char)
    
    #we add the potentially new characters present in every last measure, 
    #as we didn't go through them in the previous loop.
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

# Vectorization of the data

target_characters = sorted(list(target_characters))
num_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0

# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, num_tokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, num_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
# Save model
model.save("s2s")
