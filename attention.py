
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class Attention(Layer):

    def __init__(self, units):
        super().__init__()

        # Learnable layers for attention
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):

        # Process image features
        features = self.W1(features)

        # Process previous LSTM hidden state
        hidden = tf.expand_dims(hidden, 1)
        hidden = self.W2(hidden)

        # Calculate attention scores
        score = tf.nn.tanh(features + hidden)

        # Convert scores into attention weights
        attention_weights = tf.nn.softmax(
            self.V(score),
            axis=1
        )

        # Calculate context vector
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# Run test only when this file is executed directly
if __name__ == "__main__":

    attention = Attention(units=256)

    features = tf.random.normal((1, 49, 2048))
    hidden = tf.random.normal((1, 256))

    context_vector, attention_weights = attention(
        features, hidden
    )

    print("Context vector shape:", context_vector.shape)
    print("Attention weights shape:", attention_weights.shape)

    print(
        "Sum of attention weights:",
        tf.reduce_sum(attention_weights).numpy()
    )
