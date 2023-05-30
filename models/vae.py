import keras
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Flatten
from keras.models import Model
from keras import backend as K
from keras.losses import categorical_crossentropy



# Define Variational Autoencoder (VAE) class


class VariationalAutoEncoder(keras.Model):
    def __init__(self, input_dim, encoding_dim, learning_rate):
        super(VariationalAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = encoding_dim
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.build_model()   
        #self.vae.compile(optimizer=self.optimizer, loss=self.vae_loss)


    def build_model(self):
        # Encoder
        self.inputs = Input(shape=(self.input_dim,))
        self.encoded = Dense(128, activation="relu")(self.inputs)
        self.encoded = Dense(64, activation="relu")(self.encoded)
        self.encoded = Dense(32, activation="relu")(self.encoded)
        self.encoded = Flatten()(self.encoded)

        # Latent space
        self.z_mean = Dense(self.latent_dim)(self.encoded)
        self.z_log_var = Dense(self.latent_dim)(self.encoded)

        def sampling(args):
            z_mean, z_log_var = args
            
            epsilon = K.random_normal(shape=K.shape(z_mean), mean=0, stddev=1)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        self.z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

        self.encoder = Model(self.inputs, self.z, name='encoder')

        # Decoder
        self.decoder_input = keras.Input(shape=(self.latent_dim,))
        self.decoded = Dense(32, activation="relu")(self.decoder_input)
        self.decoded = Dense(64, activation="relu")(self.decoded)
        self.decoded = Dense(128, activation="relu")(self.decoded)
        self.decoded_out = Dense(self.input_dim, activation="sigmoid")(self.decoded)

        # Define separate decoder model
        self.decoder = Model(self.decoder_input, self.decoded_out, name="decoder")
        self.vae = Model(self.inputs, self.decoder(self.encoder(self.inputs)))
        self.vae.compile(optimizer=self.optimizer, loss=self.vae_loss)
        self.vae.summary()
        


    def vae_loss(self, x, reconstructed):
        reconstruction_loss = K.sum(K.square(x - reconstructed))
        # compute the KL loss
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.square(K.exp(self.z_log_var)), axis=-1)
        # return the average loss over all 
        total_loss = K.mean(reconstruction_loss + kl_loss)    
        #total_loss = reconstruction_loss + kl_loss
        return total_loss
    


    def train(self, x_train, epochs,  batch_size, validation_split=0.2):
        """"
        Trains the model to reconstruct the input data as precisely as possible.

        Args:
            x_train: the training data that will have to be reconstructed
            epochs: number of times x_train is passed to the network
            batch_size: the size of the sub-samples of data passed to the network in one step
            validation_split: proportion of data to be destined for training
            and proportion of data to be used for validation
        """
        #x_train = tf.convert_to_tensor(x_train)
        #x = self.decoder(self.encoder(x_train))
        #print(x.shape)
        
        #steps_per_epoch = x_train.shape[0] // batch_size
        return self.vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
                        shuffle=True, validation_split=validation_split)
    
    def check_nan_inf(self, tensor):
        """
        Checks if a tensor contains any NaN or infinity values.

        Args:
            tensor: Input tensor.

        Returns:
            True if the tensor contains NaN or infinity values, False otherwise.
        """
        return tf.reduce_any(tf.math.logical_or(tf.math.is_nan(tensor), tf.math.is_inf(tensor)))

