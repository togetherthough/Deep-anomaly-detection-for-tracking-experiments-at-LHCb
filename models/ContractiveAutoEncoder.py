# utils
import statistics
import numpy as np

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import roc_curve, auc
from models.NeuralNet_AnomalyDetection import NeuralNet_AnomalyDetection
from tensorflow.python.framework.ops import disable_eager_execution




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
            tf.keras.layers.Dense(self.input_dim, activation="sigmoid")
        ])

        self.compile(optimizer=self.optimizer, loss=self.loss)

    def encode_decode(self, inp: tf.Tensor):
        x_hid = self.encoder(inp)
        reconstructed = self.decoder(x_hid)
        return reconstructed, x_hid

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

    @tf.function
    def train(self, x_train, num_epochs, batch_size):
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

        return epoch_loss

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

        reconstructions, _ = self.encode_decode(x_train)
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

        reconstructions, _ = self.encode_decode(x_train)
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
        predictions, _ = self.encode_decode(x_test)
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
