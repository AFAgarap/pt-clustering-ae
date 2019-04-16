"""TensorFlow 2.0 implementation of variational autoencoder"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import numpy as np
import tensorflow as tf

class Sampling(tf.keras.layers.Layer):
	def call(self, args):
		z_mean, z_log_var = args
		batch = tf.shape(z_mean)[0]
		dim = tf.shape(z_mean)[1]
		epsilon = tf.random.normal(shape=(batch, dim), mean=0., stddev=1.)
		return z_mean + epsilon * tf.exp(0.5 * z_log_var)

	
class Encoder(tf.keras.layers.Layer):
	def __init__(self, intermediate_dim, latent_dim):
		super(Encoder, self).__init__()
		self.hidden_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
		self.z_mean_layer = tf.keras.layers.Dense(units=latent_dim)
		self.z_log_var_layer = tf.keras.layers.Dense(units=latent_dim)
		self.sampling = Sampling()
	
	def call(self, input_features):
		x = self.hidden_layer(input_features)
		z_mean = self.z_mean_layer(x)
		z_log_var = self.z_log_var_layer(x)
		z = self.sampling((z_mean, z_log_var))
		return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):
	def __init__(self, intermediate_dim, original_dim):
		super(Decoder, self).__init__()
		self.hidden_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
		self.output_layer = tf.keras.layers.Dense(units=original_dim, activation=tf.nn.sigmoid)

	def call(self, input_features):
		x = self.hidden_layer(x)
		output = self.output_layer(x)
		return output

class VariationalAutoencoder(tf.keras.Model):
	def __init__(self, intermediate_dim, latent_dim, original_dim):
		super(VariationalAutoencoder, self).__init__()
		self.encoder = Encoder(intermediate_dim=intermediate_dim, latent_dim=latent_dim)
		self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)
	
	def call(self, input_features):
		z_mean, z_log_var, latent_code = self.encoder(input_features)
		reconstructed = self.decoder(latent_code)
		kl_divergence = -5e-2 * tf.reduce_sum(tf.exp(z_log_var) + tf.square(z_mean) - 1 - z_log_var)
		self.add_loss(kl_divergence)
		return reconstructed
