import load_data
import vectorize_data
import lstm_model

if __name__=="__main__":

    target_tokens, max_tokens = load_data.get_tokens()

    vectorize_data.vectorize_training(target_tokens, max_tokens)

    num_tokens = len(target_tokens)

    #lstm_model.train_lstm(encoder_input_data, decoder_input_data, decoder_target_data, num_tokens)