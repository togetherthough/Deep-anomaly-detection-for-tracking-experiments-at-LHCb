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
        x = layers.Dense(16, activation="relu")(encoder_inputs)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Resample()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(16, activation="relu")(latent_inputs)
        x = layers.Dense(30, activation="relu")(x)
        decoder_outputs = layers.Dense(self.input_dim, activation="sigmoid")(x)
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


    def find_threshold_deviations(self, x_train, deviations):
        """"
        This method finds the threshold for anomaly detection as
        function of the mean and standard residual between reconstructed training data
        and the training data itself.

        Args:
            x_train: the sample of features used for training
            deviations: the number of deviations to be considered

        Returns:
            The threshold for anomaly detection.

        """

        reconstructions = self.predict(x_train)
        # find the MSE loss between the training data and the reconstructed ones
        errors = np.mean(np.power(reconstructions - x_train, 2), axis=1)
        # finally find the threshold for the
        threshold = np.mean(errors) + deviations * np.std(errors)
        return threshold

    def find_treshold_percentile(self, x_train, percentile):
        """"
        This method finds the threshold for anomaly detection as
        a percetile of the residual between reconstructed training data and
        the training data itself.

        Args:
            x_train: the sample of features used for training
            percentile: the desiered percentile to be looked up in the reconstruction errors

        Returns:
            The threshold for anomaly detection.

        """

        reconstructions = self.predict(x_train)
        # find the MSE loss between the training data and the reconstructed ones
        errors = np.mean(np.power(reconstructions - x_train, 2), axis=1)
        # set the threshold based on the percentile of the reconstruction error
        threshold = np.percentile(errors, percentile)

        return threshold

    def find_anomalies(self, x_test, threshold):
        """"
        This method makes predicitons in the forms of 1 or 0.
        Whenever the reconstruction error for one instance is bigger
        than the threshold that is classified as an anomaly, else as normal data.

        Args:
            x_test: the sample of features kept for validation
            threshold: the threshold used to classify new instances as anomaly or not

        Returns:
            This method returns 1 for anomaly and 0 for normal instance

        """
        # TODO: check if the MSE is the correct error func for this case
        predictions = self.predict(x_test)
        errors = np.mean(np.power(predictions - x_test, 2), axis=1)
        predictions = np.where(errors > threshold, 1, 0)
        return predictions

    def find_best_percentile_threshold(self, x_train, x_test, y_test):
        """"
        This method finds the best threshold for anomaly detection as
        a percetile of the residual between reconstructed training data and
        the training data itself. The metric used to score what threshold is the best is accuracy.


        Args:
            x_train: the sample of features used for training
            x_test: the sample of features used for testing
            y_test: the sample of target values kept for testing


        Returns:
            The best threshold that could be found using the percentile method.

        """
        # determining a set of suitable values to use
        # as percentiles for the "find_threshold_percentile" method (1-100 range)
        thresholds = list(range(100))
        best_threshold = None
        best_score = 0.0

        for threshold in thresholds:
            threshold = self.find_treshold_percentile(x_train, threshold)
            predictions = self.find_anomalies(x_test, threshold)

            # this is the metric to be maximized
            # (the choice is not random, since
            # given high class imbalance choosing accuracy
            # for ex. would not be suited for an anomaly detection task)
            fpr, tpr, _ = roc_curve(y_test, predictions)
            score = auc(fpr, tpr)

            # if the AUC score given by current threshold
            # is the best so far, return currrent threshold
            # as the final one to use for anomaly detection
            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def find_best_deviations_threshold(self, x_train, x_test, y_test):
        """"
        This method finds the best threshold for anomaly detection as
        a function of the mean and standard error  between reconstructed training data and
        the training data itself.
        The metric used to score what threshold is the best is accuracy.


        Args:
            x_train: the sample of features used for training
            x_test: the sample of features used for testing
            y_test: the sample of target values kept for testing


        Returns:
            The best threshold that could be found using the deviations method.

        """
        # determining a set of suitable values to use
        # as multipliers of the std in the "find_threshold_deviations" method
        thresholds = np.linspace(0.1, 3.0, num=30)
        best_threshold = None
        best_score = 0.0

        for threshold in thresholds:
            # for the desired threshold determine which are the anomalous
            # instances in the testing set
            threshold = self.find_threshold_deviations(x_train, threshold)
            predictions = self.find_anomalies(x_test, threshold)

            # this is the metric to be maximized
            fpr, tpr, _ = roc_curve(y_test, predictions)
            score = auc(fpr, tpr)

            # if the AUC score given by current threshold
            # is the best so far, return currrent threshold
            # as the final one to use for anomaly detection
            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold
