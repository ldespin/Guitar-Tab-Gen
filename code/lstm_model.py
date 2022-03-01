import numpy as np
import keras
import tensorflow as tf
import os
import re
import matplotlib.pyplot as plt
import datetime


def train_lstm(training_generator, num_tokens, epochs=30, latent_dim=256):

    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=(None, num_tokens), name="input_encoder")
    encoder = keras.layers.LSTM(latent_dim, return_state=True, name="lstm_encoder")
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.Input(shape=(None, num_tokens), name="input_decoder")

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="lstm_decoder")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_tokens, activation="softmax", name="dense_1")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="model")

    log_dir = "logs/fit_lstm_distorted/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit_generator(
        generator = training_generator,
        epochs=epochs,
        callbacks=[tensorboard_callback]
    )
    # Save model
    model.save("s2s_distorted0")