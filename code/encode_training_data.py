import numpy as np

def max_tokens(texts):
    return np.max([len(text) for text in texts])

def encode_data_lstm(input_texts, target_texts, target_tokens):

    num_tokens = len(target_tokens)
    max_encoder_seq_length = max_tokens(input_texts)
    max_decoder_seq_length = max_tokens(target_texts)

    max_encoder_file = open("../tokens/max_encoder_seq_length.txt","w")
    max_encoder_file.write(str(max_encoder_seq_length))
    max_encoder_file.close()

    token_index = dict([(token, i) for i, token in enumerate(target_tokens)])
    print(token_index)
    
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
        for t,token_input in enumerate(input_text):
            encoder_input_data[i, t, token_index[token_input]] = 1.0
        for t,token_target in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, token_index[token_target]] = 1.0
            if t>0:
                decoder_target_data[i, t-1, token_index[token_target]] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data
