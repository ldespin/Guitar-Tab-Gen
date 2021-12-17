from load_data import load_test
import vectorize_data
from tensorflow import keras

if __name__=="__main__":
    input_texts, target_texts, headers = load_test()
    tokens = list(open("../tokens/tokens_list.txt"))
    max_encoder_seq_length = int(list(open("../tokens/max_encoder_seq_length.txt"))[0])
    encoded_inputs = vectorize_data.vectorize_training(input_texts, tokens, max_encoder_seq_length)
    model = keras.models.load_model("s2s")
    generated_data_encoded = model.predict([encoded_inputs])
    generated_data_decoded = vectorize_data.unvectorize_test(generated_data_encoded, tokens)

    for i in range(len(target_texts)):
        generated = open(f"../results/{i}_generated.txt","w")
        original = open(f"../results/{i}_original.txt","w")
        for line in headers[i]:
            generated.write(line)
            original.write(line)
        for line in generated_data_decoded[i]:
            generated.write(line)
        for line in target_texts[i]:
            original.write(line)
        generated.write("end\n")
        original.write("end\n")
        generated.close()
        original.close()


