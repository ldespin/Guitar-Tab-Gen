from load_test_data import load_data
import encode_test_data
import decode_data
from tensorflow import keras

if __name__=="__main__":
    input_texts, target_texts = load_data()
    tokens = list(open("../tokens/tokens_list.txt"))
    max_encoder_seq_length = int(list(open("../tokens/max_encoder_seq_length.txt"))[0])
    encoded_inputs = encode_test_data.encode_data(input_texts, tokens, max_encoder_seq_length)
    model = keras.models.load_moedl("s2s")
    generated_data_encoded = model.predict(encoded_inputs)
    generated_data_decoded = decode_data.decode_data(generated_data_encoded, tokens)

    for i in range(len(target_texts)):
        generated = open(f"../results/{i}_generated.txt","w")
        original = open(f"../results/{i}_original.txt","w")
        for line in generated_data_decoded:
            generated.write(line)
        for line in target_texts:
            original.write(line)
        generated.close()
        original.close()


