import time
import numpy as np
from sklearn.metrics import auc, roc_curve
import tensorflow as tf
import tensorflow as tf
from keras import layers

from models.NeuralNet_AnomalyDetection import NeuralNet_AnomalyDetection


class Autoencoder(NeuralNet_AnomalyDetection):

    """"
    This is a stacked AutoEncoder class: the model here defined is an
    stacked Autoencoder made of n susequent encoders and n subsequent decoders.
    The encoder and decoder networks are themselves declared as models in here.

    The anomaly detection task in here consists in:
        1. Learning an encoding of the normal data such that the residual of the decoder
        when reconstructing such data against the actual instances is also minimized.
        2. Passing some anomalous data to the model assuming its distribution to be different
        from the normal one
        3. Find a threshold for the reconstruction erros such that the normal data have a RE
        lower than the threshold and the anomalous data has a RE larger than it.
    """

    def __init__(self, input_dim, encoding_dim, learning_rate):
        """"
        Instantiates the model.

        Args:
            input_dim: the number of features in the data
            encoding_dim: the size of the latent space,
                            to which the input will be compressed
            learning_rate: parameter that big of a step the optimizer takes
                            when optimizing over the loss function
        """

        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.model_name = "AE"

        self.build_model()

    def build_model(self):
        """"
        Builds the model as a Stacked AutoEncoder.
        The layers in both the encoder and the decoder
        are a sequence of "Dense" and "Dropout" layers.

        Dense layers: Each neuron in the dense layer receives input from each neuron of the previous layer.
        For n number of inputs (previous layer nodes) and m outputs (nodes in this layer) there will be nxm weights.
        These can be seen as a matrix W, and instead the values for the input layer can be seen as a vector x. 
        Such that the weighted sum for the inputs to each neuron is given by z = Wx. The value of each neuron will also be dependent
        on one more factor the "bias" b. Biases are also learned and added to z like this z = Wx + b.
        Finally the activation function chosen here as introduces non-linearity (doesn't produce a straight line when plotted).
        For z = Wx + b, the activation function f(z) can be summarized as f(z) = max(0, z). 
        The fact it's non-linear is good because NN can then learn non-linear functions as well.
        Dense layers can model any mathematical function. 

        Dropout layers: Dropout means that all the forward and backwards connections with a node are dropped.
        A layer with probability x implies that x% of the nodes connections will be dropped in the forward passing.
        Common technique to reduce overfitting.

        """

        # Defining an Encoder:
        # the insput shape is determined by the input data
        # of our input, such that the x's are None meaning that we can pass
        # a sample of arbitrary size, while the y's are insted the number of features.
        # The dimension of teh output for the last layer corresponds to the latent dimension.
        # As the space in which the inputs are encoded corresponds to the latent space.

        self.encoder = tf.keras.Sequential(
            # I think I'm fucking it up here....
            # the encoder directly maps the input that has few dimensions
            # to a larger space "512" and just then proceeds to compress it
            # I belive this can be an error
            [
                layers.Dense(512, activation="relu",
                             input_shape=(self.input_dim, )),
                layers.Dropout(0.1),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.1),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.1),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.1),
                layers.Dense(32, activation="relu"),
                layers.Dropout(0.1),
                layers.Dense(16, activation="relu"),
                layers.Dropout(0.1),
                layers.Dense(8, activation="relu"),
                layers.Dropout(0.1),
                layers.Dense(
                    self.encoding_dim, activation="relu", name="bottleneck")
            ])

        # Defining an Decoder:
        # The encoder takes its input from the latent space.
        # It follows its input has dimensions (None, latent dimension).
        # It learns to reconstruct the input so it progressively learns
        # to map the input to higher dimensional spaces. Its architecture is
        # specular to the one of the Encoder.
        # The output will have the shape of the Encoder's input.

        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(
                self.input_dim, activation="sigmoid", name="final")
        ])

        # instantiate an "ADAM" optimizer
        # combines Adagrad and RMSprop optimizers, under normal assumptions
        # it has been shown to perform better than other techniques
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # compiling the model sets it up for training
        # with a loss function to optimize and an optimizer object
        self.compile(
            optimizer=optimizer, loss="mse")

    def call(self, data):
        """"
        Encodes the data and then decodes it.

        Args:
            data: the input data
        Returns:
            x: reconstructed data
            x_hid: encoded data
        """

        x_hid = self.encoder(data)
        x = self.decoder(x_hid)
        return x

    def train(self, x_train, epochs,  batch_size, validation_split):
        """"
        Trains the model to reconstruct the input data as precisely as possible.

        Args:
            x_train: the training data that will have to be reconstructed
            epochs: number of times x_train is passed to the network
            batch_size: the size of the sub-samples of data passed to the network in one step
            validation_split: proportion of data to be destined for training
            and proportion of data to be used for validation
        """
        start_time = time.time()
        history = self.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
                        shuffle=True, validation_split=validation_split)

        end_time = time.time()
        training_time = end_time - start_time

        return history, training_time

    def encode(self, x_train):
        return self.encoder.predict(x_train)

    def decode(self, encoded):
        return self.decoder.predict(encoded)
