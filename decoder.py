import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Concatenate,
    Reshape
)

from tensorflow.keras.models import Model
from attention import Attention

# Project parameters
vocab_size = 8782
embedding_dim = 256
lstm_units = 256

# Inputs
image_features = Input(shape=(49, 2048))
previous_word = Input(shape=(1,), dtype="int32")

hidden_state = Input(shape=(lstm_units,))
cell_state = Input(shape=(lstm_units,))

# Attention mechanism
attention = Attention(lstm_units)

context_vector, attention_weights = attention(
    image_features,
    hidden_state
)

# Convert previous word into a vector
embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim
)

word_embedding = embedding_layer(previous_word)

# Add a time dimension to context vector
context_vector = Reshape((1, lstm_units))(context_vector)

# Combine visual and language information
decoder_input = Concatenate(axis=-1)(
    [context_vector, word_embedding]
)

# LSTM decoder
lstm_layer = LSTM(
    lstm_units,
    return_sequences=True,
    return_state=True
)

lstm_output, new_hidden, new_cell = lstm_layer(
    decoder_input,
    initial_state=[hidden_state, cell_state]
)

# Predict the next word
output_layer = Dense(vocab_size, activation="softmax")
word_prediction = output_layer(lstm_output)

# Create decoder model
decoder_model = Model(
    inputs=[
        image_features,
        previous_word,
        hidden_state,
        cell_state
    ],
    outputs=[
        word_prediction,
        new_hidden,
        new_cell,
        attention_weights
    ]
)

decoder_model.summary()