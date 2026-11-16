
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)
from tensorflow.keras.preprocessing.image import img_to_array

from decoder import decoder_model


# Load model components only once
decoder_model.load_weights("decoder.weights.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

cnn_model = ResNet50(
    weights="imagenet",
    include_top=False
)

print("Caption generation components loaded successfully!")


# Caption generation function
def generate_caption(image):

    image = image.resize((224, 224))

    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = cnn_model.predict(img_array, verbose=0)
    features = features.reshape(1, 49, 2048)

    hidden_state = tf.zeros((1, 256))
    cell_state = tf.zeros((1, 256))

    word = tokenizer.word_index["<start>"]
    caption = []

    for step in range(37):

        previous_word = tf.constant(
            [[word]], dtype=tf.int32
        )

        predictions, hidden_state, cell_state, _ = decoder_model(
            [
                tf.convert_to_tensor(features, dtype=tf.float32),
                previous_word,
                hidden_state,
                cell_state
            ],
            training=False
        )

        word_probs = predictions[0, 0, :].numpy().copy()

        # Prevent repeated two-word combinations
        if len(caption) >= 1:

            previous_caption_word = caption[-1]

            for candidate_id, candidate_word in tokenizer.index_word.items():

                for i in range(len(caption) - 1):

                    if (
                        caption[i] == previous_caption_word
                        and caption[i + 1] == candidate_word
                    ):
                        word_probs[candidate_id] = 0
                        break

        predicted_id = int(np.argmax(word_probs))

        if predicted_id == tokenizer.word_index["<end>"]:
            break

        predicted_word = tokenizer.index_word.get(
            predicted_id, "<unk>"
        )

        caption.append(predicted_word)
        word = predicted_id

    return " ".join(caption)

