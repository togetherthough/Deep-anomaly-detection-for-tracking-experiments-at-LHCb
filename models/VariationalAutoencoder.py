import time
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.NeuralNet_AnomalyDetection import NeuralNet_AnomalyDetection


class Resample(layers.Layer):

    def call(self, inputs):
        mean, log_var = inputs
        e = tf.keras.backend.random_normal(shape=(tf.shape(mean)[0], tf.shape(mean)[1]))
        return mean + tf.exp(0.5 * log_var) * e


class VariationalAutoEncoder(NeuralNet_AnomalyDetection):
    def __init__(self, input_dim, latent_dim, learning_rate):
        super(VariationalAutoEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.model_name = "VAE"
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        encoder_inputs = keras.Input(shape=(self.input_dim,))
        x = keras.layers.Dense(512, activation=tf.nn.relu)(encoder_inputs)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(256, activation=tf.nn.relu)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(128, activation=tf.nn.relu)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(64, activation=tf.nn.relu)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(32, activation=tf.nn.relu)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(16, activation=tf.nn.relu)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(8, activation=tf.nn.relu, name= "bottleneck")(x)
        x = keras.layers.Dropout(0.1)(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Resample()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = keras.layers.Dense(16, activation=tf.nn.relu)(latent_inputs)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(32, activation=tf.nn.relu)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(64, activation=tf.nn.relu)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(128, activation=tf.nn.relu)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(256, activation=tf.nn.relu)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(512, activation=tf.nn.relu)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = layers.Dense(30, activation="relu")(x)
        decoder_outputs = layers.Dense(self.input_dim, activation="sigmoid", name="final")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        self.compile(optimizer=self.optimizer)

    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
        ]
    
    def encode(self, data):
        _, _, z = self.encoder(data)
        return z

    def call(self, inp):
        _, _, z = self.encoder(inp)
        return self.decoder(z)
    
    def vae_loss(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(data, reconstruction))

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        return reconstruction_loss + kl_loss

    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            total_loss = self.vae_loss(data)
        
        #apply the gradient to the optimizer to improve the model
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)

        return {
            "loss": self.total_loss_tracker.result(),}
    

    def grad(self, data): 
        with tf.GradientTape() as tape:
            total_loss = self.vae_loss(data)
        
        #apply the gradient to the optimizer to improve the model
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return total_loss       


    def train_manual(self, data, epochs, batch_size, validation_split=0.2):
                
        # splitting the data into training and validation set
        x_train, x_val = train_test_split(data, test_size=validation_split)

        # for each epoch...
        epoch_loss = []
        start_time = time.time()
        for epoch in range(epochs):

            for x in range(0, len(x_train), batch_size):
                # the data is splitted into batches
                # of size batch data and converted to a tensor
                x_inp = tf.convert_to_tensor(
                    x_train[x: x + batch_size], dtype=tf.float32)
                # the loss for this batch of data is computed and the network updated
                loss_value = self.grad(x_inp)


            # the validation set is not used for training
            # at every epoch it will be computed the method loss on it
            x_val = tf.convert_to_tensor(x_val)
            val_loss = self.vae_loss(x_val)
            print("Epoch: {}, Train Loss: {}, Val Loss: {}".format(
                epoch, tf.reduce_mean(loss_value), tf.reduce_mean(val_loss)))

            # training loss and validation loss are concated in the epoch's loss
            epoch_loss.append(
                [tf.reduce_mean(loss_value),  tf.reduce_mean(val_loss)])


        #stop the timer and compute the difference between the start and end 
        end_time = time.time()
        training_time = end_time - start_time

        return epoch_loss, training_time
    
    def train(self, data, epochs, batch_size, validation_split=0.2):

        start_time = time.time()

        history = self.fit(data, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        #stop the timer and compute the difference between the start and end 
        end_time = time.time()
        training_time = end_time - start_time

        return history, training_time