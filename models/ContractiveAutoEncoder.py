# utils
import statistics
import time
import numpy as np

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import roc_curve, auc
from models.NeuralNet_AnomalyDetection import NeuralNet_AnomalyDetection




class ContractiveAutoEncoder(NeuralNet_AnomalyDetection):

    """"
        This is a stacked AutoEncoder class. The architecture is the same as in the
        naive AutoEncoder. The same number of layers, same types and hyperparameters.
        This is done to be able later to compare the results with higher accuracy,
        and secondly to have the opportunity to perform transfer learning between
        the different instances of AutoEncoders. 
    """

    def __init__(self, input_dim, encoding_dim, lambda_, learning_rate):
        """"
        Initializes the class.

        Args:
            input_dim: the number of features in the data
            encoding_dim: the size of the latent space,
                            to which the input will be compressed
            lambda_: is a regularalization parameter, controls the strength of the
                        contractive loss
            learning_rate: parameter that big of a step the optimizer takes
                            when optimizing over the loss function
        """

        super(ContractiveAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.lambda_ = lambda_
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.model_name = "CAE"
        self.build_model()

        # might wanna try something like
        # self.add(tf.keras.layers.Dense....name="bottleneck")

    def build_model(self):

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(
                512, activation=tf.nn.relu, input_shape=(self.input_dim,)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(8, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(
                self.encoding_dim, activation=tf.nn.relu, name="bottleneck")
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.input_dim, activation="sigmoid", name="final")
        ])

        self.compile(optimizer=self.optimizer, loss=self.loss)

    def encode_decode(self, inp: tf.Tensor):
        x_hid = self.encoder(inp)
        reconstructed = self.decoder(x_hid)
        return reconstructed, x_hid


    def encode(self, inp):
        inp = tf.convert_to_tensor(inp)
        return self.encoder(inp).numpy()

    def call(self, inp):
        x_hid = self.encoder(inp)
        return self.decoder(x_hid)

    def loss(self, x, x_bar, hidden):
        """"
        Computes the contractive loss.
        Measures the sensitivity of the encoder's output
        on small changes in the input. Computing the Jacobian 
        matrix of the last encoder's layer against the input 
        it's like computing the gradient of it wrt the input.
        To quantify the magnitude of this we take the Frobenius norm
        of the Jacobian. This will serve as a penalty term for the loss.

        Args:
            x: inputs to the model
            x_bar: reconstructed inputs
            hidden: the latent representation of the data

        Returns: 
            The contractive loss.

        """

        # find the mse of the reconstructed input and the original
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x, x_bar))

        # get the weights of the final layer of the encoder
        #!!!!!!!!!!!!!!!!!!!!!!!!

        W = tf.Variable(self.encoder.get_layer('bottleneck').weights[0])
        W = tf.transpose(W)

        # this is the penalty term
        contractive = K.sum(((hidden * (1 - hidden))**2)
                            * K.sum(W**2, axis=1), axis=1)

        total_loss = reconstruction_loss + (self.lambda_ * contractive)
        return total_loss

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

            reconstruction, hidden = self.encode_decode(inputs)

            loss_value = self.loss(inputs, reconstruction, hidden)

        return loss_value, tape.gradient(loss_value, self.trainable_variables), reconstruction


    def train(self, x_train, epochs, batch_size, validation_split=0.2):
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
        x_train, x_val = train_test_split(x_train, test_size=validation_split)

        # keeps the count of the current global training step
        global_step = tf.Variable(0)

        # for each epoch...
        epoch_loss = []
        start_time = time.time()
        for epoch in range(epochs):

            for x in range(0, len(x_train), batch_size):
                # the data is splitted into batches
                # of size batch data and converted to a tensor
                x_inp = tf.convert_to_tensor(
                    x_train[x: x + batch_size], dtype=tf.float32)
                # the loss for this batch of data is computed
                # and returned along side its gradient
                # a gradient is the direction and magnitude
                # of the steepest descent of the loss function from this point
                loss_value, grads, _ = self.grad(x_inp)
                # the gradient is passed to the optimizer alongside the methods
                # trainable varibles so that these can be adjusted according to the grad
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables),
                                               global_step)

            # the validation set is not used for training
            # at every epoch it will be computed the method loss
            # on it
            reconstructed, hidden = self.encode_decode(x_val)
            x_val = tf.convert_to_tensor(x_val)
            val_loss = self.loss(x_val, reconstructed, hidden)
            print("Step: {}, Train Loss: {}, Val Loss: {}".format(
                global_step.numpy(), tf.reduce_mean(loss_value), tf.reduce_mean(val_loss)))

            # training loss and validation loss are concated in the epoch's loss
            epoch_loss.append(
                [tf.reduce_mean(loss_value),  tf.reduce_mean(val_loss)])
        
        #stop the timer and compute the difference between the start and end 
        end_time = time.time()
        training_time = end_time - start_time

        return epoch_loss, training_time
