
from sklearn.metrics import auc, roc_curve
import tensorflow as tf
from keras.initializers import glorot_normal
# ml
from tensorflow_addons.layers import SpectralNormalization
import tensorflow as tf

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