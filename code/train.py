from load_training_data import load_data
import encode_training_data
import lstm_model

if __name__=="__main__":

    input_train_texts, target_train_texts, target_tokens = load_data()

    encoder_input_data, decoder_input_data, decoder_target_data = encode_training_data.encode_data_lstm(input_train_texts, target_train_texts, target_tokens)

    num_tokens = len(target_tokens)

    lstm_model.train_lstm(encoder_input_data, decoder_input_data, decoder_target_data, num_tokens)