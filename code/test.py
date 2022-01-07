from load_data import load_test
import vectorize_data
from tensorflow import keras
import numpy as np

if __name__=="__main__":
    input_texts, target_texts, headers = load_test()
    tokens = list(open("../tokens/tokens_list.txt"))
    num_tokens = len(tokens)
    max_encoder_seq_length = int(list(open("../tokens/max_encoder_seq_length.txt"))[0])
    encoded_inputs = vectorize_data.vectorize_test(input_texts, tokens, max_encoder_seq_length)
    model = keras.models.load_model("s2s")

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states, name="encoder")

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = keras.Input(shape=(256,))
    decoder_state_input_c = keras.Input(shape=(256,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)

    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states, name="decoder"
    )

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_token_index = dict((i, token) for token, i in enumerate(tokens))

    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_tokens))

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_token_index[sampled_token_index]
            decoded_sentence += sampled_token

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_token == "\n" or len(decoded_sentence) > max_encoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]
        return decoded_sentence

    for seq_index in range(len(encoded_inputs)):
        input_seq = encoded_inputs[seq_index]
        decoded_measure = decode_sequence(input_seq)
        print("-")
        print("Input measure:", input_texts[seq_index])
        print("Decoded measure:", decoded_measure)


