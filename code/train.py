from load_data import load_training
import vectorize_data
import lstm_model

if __name__=="__main__":

    input_train_texts, target_train_texts, target_tokens = load_training()

    encoder_input_data, decoder_input_data, decoder_target_data = vectorize_data.vectorize_training(input_train_texts, target_train_texts, target_tokens)

    num_tokens = len(target_tokens)

    lstm_model.train_lstm(encoder_input_data, decoder_input_data, decoder_target_data, num_tokens)