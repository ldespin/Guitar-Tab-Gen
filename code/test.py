from load_data import load_test
import vectorize_data
from tensorflow import keras
import numpy as np
import os

if __name__=="__main__":

    input_texts = []
    target_texts = []

    measures = os.listdir("../training_data/raw")
    n=999
    IDs = []
    for i in range(999):
        song = []
        for file in measures:
            if f"_{i}_" in file:
                song.append(file)
        for j in range(len(song)-1):
            IDs.append([i,j])

    for i, ID in enumerate(IDs[:-1]):
            # Store sample
            input_file = open(f'../training_data/raw/data_{ID[0]}_{ID[1]}.txt')
            input_text = list(input_file)

            target_file = open(f'../training_data/raw/data_{ID[0]}_{ID[1]+1}.txt')
            target_text = list(target_file)

            input_texts.append(input_text)
            target_texts.append(target_text)

    tokens = list(open("../tokens/tokens_list.txt"))
    num_tokens = len(tokens)
    max_encoder_seq_length = int(list(open("../tokens/max_encoder_seq_length.txt"))[0])
    model = keras.models.load_model("s2s")

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states, name="encoder")

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = keras.Input(shape=(256,),name="input_state_h")
    decoder_state_input_c = keras.Input(shape=(256,),name="input_state_c")
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
    reverse_token_index = dict((i, token) for i, token in enumerate(tokens))
    token_index = dict((token, i) for i, token in enumerate(tokens))

    def get_duration(sequence):
        dur = 0
        for line in sequence:
            if "wait" in line:
                dur+=int(line.split(':')[1])
        return dur

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def decode_sequence(input_seq,dur_max):
        first_token = True

        #Check if the laste generated token as a wait token
        wait_generated = False

        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Duration of the generated mesure
        total_gen_dur = 0

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_tokens))

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token

            # Ban wait if we generated one previously

            #if wait_generated or first_token:
            #    for token in tokens:
            #        if "wait" in token:
            #            output_tokens[0,-1,token_index[token]]=0
            #    first_token = False

            sampled_token_index = sample(output_tokens[0, -1, :])
            sampled_token = reverse_token_index[sampled_token_index]

            
            #We control the duration of the measure, comparing it to the duration of the input one.
            if "wait" in sampled_token:
                #If we generate a wait token that doesn't reach the max duration, we add it to the generated measure
                gen_dur=int(sampled_token.split(':')[1])
                total_gen_dur+=gen_dur
                if total_gen_dur<dur_max:  
                    decoded_sentence += sampled_token
                #Otherwise, we add a wait token that will end the measure (potentially wait:0)         
                else:
                    decoded_sentence+=f"wait:{dur_max-total_gen_dur}\n"
                    return decoded_sentence
                wait_generated = True
            else:
                decoded_sentence += sampled_token
                wait_generated = False

            # Exit condition: either hit max length
            # or find stop character.
            if len(decoded_sentence) > max_encoder_seq_length:
                stop_condition = True
                decoded_sentence+=f"wait:{dur_max-total_gen_dur}\n"

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]
        return decoded_sentence


    for seq_index in range(len(input_texts)):
        f_input = open(f"../results/seq_{seq_index}_input",'w')
        f_output_or = open(f"../results/seq_{seq_index}_output_or",'w')
        f_output_pred = open(f"../results/seq_{seq_index}_output_pred",'w')
        input_seq = vectorize_data.vectorize_test(input_texts[seq_index])
        dur= get_duration(input_texts[seq_index])
        decoded_measure = decode_sequence(input_seq,dur)

        for token in input_texts[seq_index]:
            f_input.write(token)

        for token in target_texts[seq_index]:
            f_output_or.write(token)

        f_output_pred.write(decoded_measure)

        f_input.close()
        f_output_or.close()
        f_output_pred.close()



