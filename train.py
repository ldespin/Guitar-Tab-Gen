from load_data import load_data
import encode_data
import lstm_model
import h5py

if __name__=="__main__":

    input_train_texts, input_test_texts, target_train_texts, target_test_texts, target_tokens = load_data()

    hf_X = h5py.File('X_test_data', 'w')
    hf_X.create_dataset('X_test', data=input_test_texts)
    hf_X.close()

    hf_Y = h5py.File('Y_test_data', 'w')
    hf_Y.create_dataset('Y_test', data=target_test_texts)
    hf_Y.close()

    encoder_input_data, decoder_input_data, decoder_target_data = encode_data.encode_data_lstm(input_train_texts, target_train_texts, target_tokens)

    num_tokens = len(target_tokens)

    lstm_model.train_lstm(encoder_input_data, decoder_input_data, decoder_target_data, num_tokens)