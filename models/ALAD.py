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


# Define the GAN architecture

# https://www.kaggle.com/code/sekfook97/gan-for-anomaly-detection

# https://www.kaggle.com/code/francoisdubois/bidirectionnal-gan-keras


#this network learns to discriminate between natually distributed noise
#and the real data, it is trained alongside the other two discriminators 

class DiscriminatorXZ(tf.keras.Model):
    def __init__(self):
        super(DiscriminatorXZ, self).__init__()
        # I'M NOT SURE THE CONCATENATE SHOULD BE WHERE IT IS NOW

        self.dense_x = tf.keras.layers.Dense(256, activation='relu')
        self.dense_z = tf.keras.layers.Dense(256, activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        """"
        The network takes as input both the 
        real data and the fake one

        Args:
            inputs: tuple made of real and fake data
        """
        x, z = inputs
        x_out = self.dense_x(x)
        z_out = self.dense_z(z)
        concatenated = self.concat([x_out, z_out])
        output = self.output_layer(concatenated)
        return output

class DiscriminatorZZ(tf.keras.Model):
    def __init__(self):
        super(DiscriminatorZZ, self).__init__()
        # I'M NOT SURE THE CONCATENATE SHOULD BE WHERE IT IS NOW

        self.dense_z1 = SpectralNormalization(tf.keras.layers.Dense(256, activation='relu'))
        self.dense_z2 = SpectralNormalization(tf.keras.layers.Dense(256, activation='relu'))
        self.concat = tf.keras.layers.Concatenate()
        self.output_layer = SpectralNormalization(tf.keras.layers.Dense(1, activation='sigmoid'))

    def call(self, inputs):
        """"
        The network takes as input both the 
        real data and the fake one

        Args:
            inputs: tuple made of real and fake data
        """
        
        z_1, z_2 = inputs       
        z1out = self.dense_z1(z_1)        
        z2out = self.dense_z2(z_2)       
        concatenated = self.concat([z1out, z2out])      
        output = self.output_layer(concatenated)
        return output

class ALAD(NeuralNet_AnomalyDetection):
    def __init__(self, input_dim, latent_dim, learning_rate):
        super(ALAD, self).__init__()
        #setting up the input and latent dimensions 
        self.input_dim = input_dim
        self.latent_dim = latent_dim
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
            tf.keras.layers.InputLayer(input_shape=(
                None, input_dim)),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_initializer=glorot_normal),
            tf.keras.layers.Dense(32, activation='relu',
                                  kernel_initializer=glorot_normal),
            tf.keras.layers.Dense(
                latent_dim, kernel_initializer=glorot_normal),
        ])

        # The generator tries to replicate the original distribution 
        # of the training data. TO guarantee cycle consistency 
        # this method aims at having G(E(x)) = x and E(G(z)) = z
        self.generator = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu',
                                  kernel_initializer=glorot_normal),
            tf.keras.layers.Dense(128, activation='relu',
                                  kernel_initializer=glorot_normal),

            tf.keras.layers.Dense(input_dim, kernel_initializer=glorot_normal),
        ])

        # GAN is vulnerable to mode collapse and training instability.
        # Spectral Normalization resolves these problems.
        # this has to be applied only on the discriminator zz

        self.discriminator_xz = DiscriminatorXZ()

        self.discriminator_xx = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu',
                                  input_shape=(None, input_dim)),

            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])

        self.discriminator_zz = DiscriminatorZZ()

    def discriminator_loss_xz(self, l_encoder, l_generator):
        """"
        Computes the loss for the xz-discriminator
        """

         # discriminator xz
        loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(l_encoder),logits=l_encoder))
        loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(l_generator),logits=l_generator))

        return loss_dis_gen + loss_dis_enc

    def discriminator_loss_xx(self, x_logit_real, x_logit_fake):
        """"
        Computes the loss for the xx-discriminator
        """
        x_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit_real, labels=tf.ones_like(x_logit_real)
        )
        x_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit_fake, labels=tf.zeros_like(x_logit_fake)
        )
        return tf.reduce_mean(x_real_dis + x_fake_dis)

    def discriminator_loss_zz(self, z_logit_real, z_logit_fake):
        """"
        Computes the loss for the zz-discriminator
        """
        z_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=z_logit_real, labels=tf.ones_like(z_logit_real)
        )
        z_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=z_logit_fake, labels=tf.zeros_like(z_logit_fake)
        )
        return tf.reduce_mean(z_real_dis + z_fake_dis)


    def encoder_generator__loss(self, l_generator, l_encoder,
                     x_logit_real, x_logit_fake, z_logit_real, z_logit_fake):
        

        labels_gen = tf.zeros_like(l_generator)
                    # Labels for the encoder (fake representations)
        labels_enc = tf.ones_like(l_encoder)

        gen_loss_xz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_gen, logits=l_generator)
        )
        enc_loss_xz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_enc, logits=l_encoder)
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

    def train(self, train_data, num_epochs, batch_size):

        # Training loop
        for epoch in range(num_epochs):
            # Shuffle the training data
            #train_data = train_data.sample(frac=1)

            # Calculate the number of batches
            num_batches = len(train_data) // batch_size

            for batch_index in range(num_batches):
                
            

                with tf.GradientTape() as gen_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as disc_xz_tape, tf.GradientTape() as disc_xx_tape, tf.GradientTape() as disc_zz_tape:
                

                    batch_real_data = train_data[batch_index *
                                             batch_size: (batch_index + 1) * batch_size]

                    
                    batch_latent_vars = np.random.normal(size=(batch_size, self.latent_dim))
                
                    # generate data from the latent representation
                    batch_fake_data = self.generator(batch_latent_vars)
                    batch_latent_vars = tf.convert_to_tensor(batch_latent_vars, dtype=tf.float32)

                    #convert the real and fake data to tensors to be passed to the discriminators
                    batch_real_data = tf.convert_to_tensor(batch_real_data, dtype=tf.float32)
                    batch_fake_data = tf.convert_to_tensor(batch_fake_data, dtype=tf.float32)

                    # Compute logits for the discriminator xx
                    x_logit_real = self.discriminator_xx(batch_real_data)
                    x_logit_fake = self.discriminator_xx(batch_fake_data)

                    #zz-discriminator for inputs (z, z) and (z, E(G(z))
                    z_logit_real = self.discriminator_zz([batch_latent_vars, batch_latent_vars])
                    z_logit_fake = self.discriminator_zz([batch_latent_vars, self.encoder(self.generator(batch_real_data))])

                    #xz-discriminator for inputs (G(z),z) and (x, E(x))
                    l_generator = self.discriminator_xz([self.generator(batch_fake_data), batch_fake_data])
                    l_encoder = self.discriminator_xz([batch_real_data, self.encoder(batch_real_data)])                 

                    
                    # Compute the losses for generator and encoder
                    enc_loss, gen_loss = self.encoder_generator__loss(
                        l_generator, l_encoder,
                        x_logit_real, x_logit_fake, z_logit_real, z_logit_fake
                    )
            
                    #compute the loss for the different discriminators 
                    disc_xz_loss = self.discriminator_loss_xz(
                        l_encoder, l_generator
                    )
                    disc_xx_loss = self.discriminator_loss_xx(
                        x_logit_real, x_logit_fake
                    )
                    disc_zz_loss = self.discriminator_loss_zz(
                        z_logit_real, z_logit_fake
                    )

                # Calculate the gradients for the different networks
                gen_gradients = gen_tape.gradient(
                    gen_loss, self.generator.trainable_variables
                )
                enc_gradients = enc_tape.gradient(
                    enc_loss, self.encoder.trainable_variables
                )
                disc_xz_gradients = disc_xz_tape.gradient(
                    disc_xz_loss, self.discriminator_xz.trainable_variables
                )
                disc_xx_gradients = disc_xx_tape.gradient(
                    disc_xx_loss, self.discriminator_xx.trainable_variables
                )
                disc_zz_gradients = disc_zz_tape.gradient(
                    disc_zz_loss, self.discriminator_zz.trainable_variables
                )

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

            print("Step: {},  Encoder Loss: {}, Decoder Loss: {}, Disc_xx Loss: {}, Disc_xz Loss: {}, Disc_zz Loss: {}".format(
                epoch, tf.reduce_sum(enc_loss), tf.reduce_sum(gen_loss), tf.reduce_sum(disc_xx_loss), tf.reduce_sum(disc_xz_loss), tf.reduce_sum(disc_zz_loss)))

    def predict(self, x):
        
        # get the encoded data
        z = self.encoder(x)  
        # generate the synthetic data
        x_hat = self.generator(z)  
        
        #finally get the difference between 
        # the discrimination value of the real data
        # and the real data versus generated one pair
        f_xx_x = self.discriminator_xx(x, x)
        f_xx_x_hat = self.discriminator_xx(x, x_hat)
        #anomaly score 
        anomaly_score = abs(f_xx_x - f_xx_x_hat)  
        return anomaly_score
