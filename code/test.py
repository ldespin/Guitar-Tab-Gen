from load_test_data import load_data
import encode_test_data
import decode_data
#from tensorflow import keras

if __name__=="__main__":
    input_texts, target_texts = load_data()
    tokens = list(open("../tokens/tokens_list.txt"))
    max_encoder_seq_length = int(list(open("../tokens/max_encoder_seq_length.txt"))[0])

    encoded_inputs = encode_test_data.encode_data(input_texts, tokens, max_encoder_seq_length)
    decoded_inputs = decode_data.decode_data(encoded_inputs, tokens)
    print(input_texts)
    print(decoded_inputs)
    #model = keras.models.load_moedl("s2s")
    #generated_data_encoded = model.predict(encoded_inputs)


