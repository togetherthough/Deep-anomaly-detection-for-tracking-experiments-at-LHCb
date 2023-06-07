import time
from sklearn.metrics import auc, roc_curve
import tensorflow as tf
import numpy as np
from keras.initializers import glorot_normal
# utils
import pandas as pd
import numpy as np
from tensorflow_addons.layers import SpectralNormalization
#from tensorflow_models.nlp import SpectralNormalization

# machine learning
from sklearn.model_selection import train_test_split
import tensorflow as tf

from models.NeuralNet_AnomalyDetection import NeuralNet_AnomalyDetection


#this network learns to discriminate between natually distributed noise
#and the real data, it is trained alongside the other two discriminators 

class DiscriminatorXZ(tf.keras.Model):
    def __init__(self, input_dim, latent_dim):

        #Discriminates between pairs (E(x), x) and (z, G(z))

        super(DiscriminatorXZ, self).__init__()
  
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.x = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            SpectralNormalization(tf.keras.layers.Dense(256, activation='relu')),
            SpectralNormalization(tf.keras.layers.Dense(512, activation='relu')),
        ])

        self.z = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.latent_dim, )),
            SpectralNormalization(tf.keras.layers.Dense(256, activation='relu')),
            SpectralNormalization(tf.keras.layers.Dense(512, activation='relu')),
        ])
        
        self.concatenate = tf.keras.layers.Concatenate()
        self.output_layer = SpectralNormalization(tf.keras.layers.Dense(1))

    def call(self, inputs):
        """"
        The network takes as input both the 
        real data and the fake one

        Args:
            inputs: tuple made of real and fake data
        """
        x, z = inputs
        x_out = self.x(x)
        z_out = self.z(z)
        concatenated = self.concatenate([x_out, z_out])
        output_val = self.output_layer(concatenated)
        return output_val



class DiscriminatorXX(tf.keras.Model):
    def __init__(self, input_dim):
        super(DiscriminatorXX, self).__init__()

        self.input_dim = input_dim
        
        self.concatenate = tf.keras.layers.Concatenate()

        self.x = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, self.input_dim + self.input_dim)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(2048, activation='relu'),
        ])
        
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """"
        The network takes as input both the 
        real data and the fake one

        Args:
            inputs: tuple made of real and fake data
        """
        
        x_1, x_2 = inputs  
        concatenated = self.concatenate([x_1, x_2])      
        x_out = self.x(concatenated)                  
        output_val = self.output_layer(x_out)
        return output_val



class DiscriminatorZZ(tf.keras.Model):
    def __init__(self, latent_dim):
        super(DiscriminatorZZ, self).__init__()

        #concatenate immediatelly here

        self.latent_dim = latent_dim

        self.concatenate = tf.keras.layers.Concatenate()

        self.z = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, self.latent_dim + self.latent_dim)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(2048, activation='relu'),
        ])
        
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """"
        The network takes as input both the 
        real data and the fake one

        Args:
            inputs: tuple made of real and fake data
        """
        
        z_1, z_2 = inputs  
        concatenated = self.concatenate([z_1, z_2])        
        z = self.z(concatenated)
        output_val = self.output_layer(z)
        return output_val

class ALAD(NeuralNet_AnomalyDetection):
    def __init__(self, input_dim, latent_dim, learning_rate):
        super(ALAD, self).__init__()
        #setting up the input and latent dimensions 
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model_name = "ALAD"

        #setting up the optimizers for every network
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.discriminator_xz_optimizer = tf.keras.optimizers.Adam(
            learning_rate)
        self.discriminator_xx_optimizer = tf.keras.optimizers.Adam(
            learning_rate)
        self.discriminator_zz_optimizer = tf.keras.optimizers.Adam(
            learning_rate)

        # The encoder learns a compressed represnetation of the data
        # it is trained alongside the generator. 
        # using the glorot normal because is the same of the
        # original code glorot = xavier, it can be found in the paper
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
                self.latent_dim, activation=tf.nn.relu, name="bottleneck")
        ])

        # The generator tries to replicate the original distribution 
        # of the training data. TO guarantee cycle consistency 
        # this method aims at having G(E(x)) = x and E(G(z)) = z
        self.generator = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu',
                                  kernel_initializer=glorot_normal),
            tf.keras.layers.Dense(128, activation='relu',
                                  kernel_initializer=glorot_normal),
            tf.keras.layers.Dense(128, activation='relu',
                                  kernel_initializer=glorot_normal),

            tf.keras.layers.Dense(self.input_dim, kernel_initializer=glorot_normal),
        ])

        # GAN is vulnerable to mode collapse and training instability.
        # Spectral Normalization resolves these problems.
        # this has to be applied only on the discriminator zz

        self.discriminator_xz = DiscriminatorXZ(self.input_dim, self.latent_dim)

        self.discriminator_xx = DiscriminatorXX(self.input_dim)

        self.discriminator_zz = DiscriminatorZZ(self.latent_dim)

    def discriminator_loss_xz(self, xz_true, xz_fake, y_true, y_fake):
        """"
        Computes the loss for the xz-discriminator
        """
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true,logits=xz_true)) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_fake,logits=xz_fake))

        return loss

    def discriminator_loss_xx(self, x_logit_real, x_logit_fake):
        """"
        Computes the loss for the xx-discriminator
        """
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit_real, labels=tf.ones_like(x_logit_real)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit_fake, labels=tf.zeros_like(x_logit_fake))
        
        return tf.reduce_mean(loss)

    def discriminator_loss_zz(self, z_logit_real, z_logit_fake):
        """"
        Computes the loss for the zz-discriminator
        """
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=z_logit_real, labels=tf.ones_like(z_logit_real)) + tf.nn.sigmoid_cross_entropy_with_logits(
            logits=z_logit_fake, labels=tf.zeros_like(z_logit_fake))
        
        return tf.reduce_mean(loss)


    def encoder_generator__loss(self, xz_fake, xz_true,
                     x_logit_real, x_logit_fake, z_logit_real, z_logit_fake):
        

        #logits are correct 

        gen_loss_xz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(xz_fake), logits=xz_fake)
        )  
        enc_loss_xz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(xz_true), logits=xz_true)
        )


        x_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit_real, labels=tf.zeros_like(x_logit_real)
        )
        x_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit_fake, labels=tf.ones_like(x_logit_fake)
        )
        z_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=z_logit_real, labels=tf.zeros_like(z_logit_real)
        )
        z_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=z_logit_fake, labels=tf.ones_like(z_logit_fake)
        )

        cost_x = tf.reduce_mean(x_real_gen + x_fake_gen)
        cost_z = tf.reduce_mean(z_real_gen + z_fake_gen)

        cycle_consistency_loss = cost_x + cost_z


        return enc_loss_xz + cycle_consistency_loss, gen_loss_xz + cycle_consistency_loss

    def train(self, train_data, epochs, batch_size, validation_split=None):

        # Training loop
        for epoch in range(epochs):
            # Shuffle the training data
            #train_data = train_data.sample(frac=1)

            # Calculate the number of batches
            num_batches = len(train_data) // batch_size

            epoch_loss = []
            start_time = time.time()

            for batch_index in range(num_batches):
                
            

                with tf.GradientTape() as gen_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as disc_xz_tape, tf.GradientTape() as disc_xx_tape, tf.GradientTape() as disc_zz_tape:
                
                    #get a batch of the data passed for training  
                    batch_real_data = train_data[batch_index *
                                             batch_size: (batch_index + 1) * batch_size]

                    #generate some data of teh same size of the lantent space
                    batch_latent_vars = np.random.normal(size=(batch_size, self.latent_dim))
                                        
                    # generate data from the latent representation
                    batch_fake_data = self.generator(batch_latent_vars)

                    #encode the original data into the latent space
                    batch_encoded_data = self.encoder(batch_real_data)


                    y_true = tf.ones((tf.shape(batch_real_data)[0], 1), dtype=tf.float32)
                    y_fake = tf.zeros((tf.shape(batch_real_data)[0], 1), dtype=tf.float32)

                    #convert all the variables 
                    batch_latent_vars = tf.convert_to_tensor(batch_latent_vars, dtype=tf.float32)
                    batch_encoded_data = tf.convert_to_tensor(batch_encoded_data, dtype=tf.float32)
                    batch_real_data = tf.convert_to_tensor(batch_real_data, dtype=tf.float32)
                    batch_fake_data = tf.convert_to_tensor(batch_fake_data, dtype=tf.float32)

                    # Compute logits for the discriminator xx
                    x_logit_real = self.discriminator_xx.call([batch_real_data, batch_real_data])
                    x_logit_fake = self.discriminator_xx.call([batch_real_data, self.generator(self.encoder(batch_real_data))])

                    #zz-discriminator for inputs (z, z) and (z, E(G(z))
                    z_logit_real = self.discriminator_zz.call([batch_latent_vars, batch_latent_vars])
                    z_logit_fake = self.discriminator_zz.call([batch_latent_vars, self.encoder(self.generator(batch_latent_vars))])

                    #can the error be here??

                    #xz-discriminator for inputs (G(z),z) and (x, E(x))
                    xz_real = self.discriminator_xz.call([batch_real_data, self.encoder(batch_real_data)])    
                    xz_fake = self.discriminator_xz.call([batch_fake_data, batch_latent_vars])
                                 

                    
                    # Compute the losses for generator and encoder
                    enc_loss, gen_loss = self.encoder_generator__loss(xz_fake, xz_real,
                        x_logit_real, x_logit_fake, z_logit_real, z_logit_fake)
            
                    #compute the loss for the different discriminators 
                    disc_xz_loss = self.discriminator_loss_xz(xz_real, xz_fake, y_true, y_fake)
                    disc_xx_loss = self.discriminator_loss_xx(x_logit_real, x_logit_fake)
                    disc_zz_loss = self.discriminator_loss_zz(z_logit_real, z_logit_fake)

                # Calculate the gradients for the different networks
                gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                enc_gradients = enc_tape.gradient(enc_loss, self.encoder.trainable_variables)
                disc_xz_gradients = disc_xz_tape.gradient(disc_xz_loss, self.discriminator_xz.trainable_variables)
                disc_xx_gradients = disc_xx_tape.gradient(disc_xx_loss, self.discriminator_xx.trainable_variables)
                disc_zz_gradients = disc_zz_tape.gradient(disc_zz_loss, self.discriminator_zz.trainable_variables)

                # update the weights of the generator, encoder, and discriminators
                # to improve the performance of each network
                self.generator_optimizer.apply_gradients(
                    zip(gen_gradients, self.generator.trainable_variables))
                self.encoder_optimizer.apply_gradients(
                    zip(enc_gradients, self.encoder.trainable_variables))
                self.discriminator_xz_optimizer.apply_gradients(
                    zip(disc_xz_gradients, self.discriminator_xz.trainable_variables))
                self.discriminator_xx_optimizer.apply_gradients(
                    zip(disc_xx_gradients, self.discriminator_xx.trainable_variables))
                self.discriminator_zz_optimizer.apply_gradients(
                    zip(disc_zz_gradients, self.discriminator_zz.trainable_variables))

            print("Epoch: {},  Encoder Loss: {}, Generator Loss: {}, Disc_xx Loss: {}, Disc_xz Loss: {}, Disc_zz Loss: {}".format(
                epoch+1, tf.reduce_sum(enc_loss), tf.reduce_sum(gen_loss), tf.reduce_sum(disc_xx_loss), tf.reduce_sum(disc_xz_loss), tf.reduce_sum(disc_zz_loss)))
            
            epoch_loss.append(
                [tf.reduce_mean(gen_loss),  tf.reduce_mean(disc_xx_loss+disc_xz_loss+disc_zz_loss)])
            
        #stop the timer and compute the difference between the start and end 
        end_time = time.time()
        training_time = end_time - start_time

        return epoch_loss, training_time

    def predict(self, x):
        
        # get the encoded data
        z = self.encoder(x)  
        # generate the synthetic data
        x_hat = self.generator(z)  
        
        #finally get the difference between 
        # the discrimination value of the real data
        # and the real data versus generated one pair
        f_xx_x = self.discriminator_xx([x, x])
        f_xx_x_hat = self.discriminator_xx([x, x_hat])
        #anomaly score 
        anomaly_score = abs(f_xx_x - f_xx_x_hat)  
        return anomaly_score
    
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
    

    def find_anomalies(self, x_test, threshold):
        """"
        This method makes predicitons in the forms of 1 or 0,
        if an instance is considered to be an anomaly its
        value will be 1, if is a normal instace its value will be 0.
        Whenever the determinant value (reconstruction error,
        anomaly score...etc) for one instance is bigger than the
        threshold that is classified as an anomaly, else as normal data.

        Args:
            x_test: the sample of features kept for validation
            threshold: the threshold used to classify new instances as anomaly or not

        Returns:
            This method returns a list containg 1 for anomalies and 0 for normal instances

        """

        predictions = self.predict(x_test)
        predictions = np.where(predictions > threshold, 1, 0)

        return predictions

