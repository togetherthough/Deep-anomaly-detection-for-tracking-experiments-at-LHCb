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

        # Latent space
        self.z_mean = Dense(self.latent_dim)(self.encoded)
        self.z_log_var = Dense(self.latent_dim)(self.encoded)

        class Sampling(Layer):
 
            def call(self, inputs):
                z_mean, z_log_var = inputs
                epsilon = tf.keras.backend.random_normal(shape =(z_mean.shape))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
        self.z = Sampling()([self.z_mean, self.z_log_var])

        self.encoder = Model(self.inputs, self.z_mean, name='encoder')

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
    
    def call(self, inp):
        encoded = self.encoder(inp)
        return self.decoder(encoded)
    
    def check_nan_inf(self, tensor):
        """
        Checks if a tensor contains any NaN or infinity values.

        Args:
            tensor: Input tensor.

        Returns:
            True if the tensor contains NaN or infinity values, False otherwise.
        """
        return tf.reduce_any(tf.math.logical_or(tf.math.is_nan(tensor), tf.math.is_inf(tensor)))


    def grad(self, inputs):
        """
        Computes the gradient of the loss, on the trainable varibles.

        Args:
            inputs: the subsample of x_train on whichs the gradient needs to be computed

        Return: 
            loss_value: the loss for input, calculated bewteen input & reconstructed version of it
            tape.gradient: the gradient giving the direction and magnitude of steepest descent
            reconstruction: reconstructed inputs

        """
        # creates a gradient tape context
        # when operations are performed in this context
        # tensorflow records the operations performed here
        # and the varibles accessed. With the two combined
        # tf builds a computation graph that can be used to
        # compute gradients
        with tf.GradientTape() as tape:
            # encode and decode the input

            reconstruction= self.call(inputs)

            loss_value = self.vae_loss(inputs, reconstruction)

        return loss_value, tape.gradient(loss_value, self.vae.trainable_variables), reconstruction




    def trainnn(self, x_train, num_epochs, batch_size):
        """"
        Trains the CAE.

        Args:
            x_train: the sample of data used for training the network
            epochs: number of times x_train is passed to the network
            batch_size: the size of the sub-samples of data passed to the network in one step

        Returns:
            epoch_loss: a list containing the NN's loss over the training set
                        and over the validation one
        """
        
        
        # splitting the data into training and validation set
        x_train, x_val = train_test_split(x_train, test_size=0.2)

        # keeps the count of the current global training step
        global_step = tf.Variable(0)

        # for each epoch...
        epoch_loss = []
        for epoch in range(num_epochs):
            print("Epoch: ", epoch)

            for x in range(0, len(x_train), batch_size):
                # the data is splitted into batches
                # of size batch data and converted to a tensor
                x_inp = tf.convert_to_tensor(
                    x_train[x: x + batch_size], dtype=tf.float32)
                # the loss for this batch of data is computed
                # and returned along side its gradient
                # a gradient is the direction and magnitude
                # of the steepest descent of the loss function from this point
                loss_value, grads = self.grad(x_inp)
                # the gradient is passed to the optimizer alongside the methods
                # trainable varibles so that these can be adjusted according to the grad
                self.optimizer.apply_gradients(zip(grads, self.vae.trainable_variables),
                                               global_step)

            # the validation set is not used for training
            # at every epoch it will be computed the method loss
            # on it
            reconstructed = self.call(x_val)
            x_val = tf.convert_to_tensor(x_val)
            val_loss = self.vae_loss(x_val, reconstructed)
            print("Step: {}, Train Loss: {}, Val Loss: {}".format(
                global_step.numpy(), tf.reduce_mean(loss_value), tf.reduce_mean(val_loss)))

            # training loss and validation loss are concated in the epoch's loss
            epoch_loss.append(
                [tf.reduce_mean(loss_value),  tf.reduce_mean(val_loss)])

        return epoch_loss