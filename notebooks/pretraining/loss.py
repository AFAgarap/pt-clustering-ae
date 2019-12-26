#!/usr/bin/env python3

import tensorflow as tf


def SNNL(features, labels, distance="euclidean", temperature=100.0):
    distance = distance.lower()

    stability_epsilon = 1e-5

    if distance == "euclidean":
        squared_norm_a = tf.reshape(
            tf.reduce_sum(tf.square(features), axis=1), [1, tf.shape(features)[0]]
        )
        squared_norm_b = tf.reshape(
            tf.reduce_sum(tf.square(features), axis=1), [tf.shape(features)[0], 1]
        )
        inner_product = tf.matmul(features, features, transpose_b=True)
        tile_a = tf.tile(squared_norm_a, [tf.shape(features)[0], 1])
        tile_b = tf.tile(squared_norm_b, [1, tf.shape(features)[0]])
        distance_matrix = tile_a + tile_b - 2.0 * inner_product
    elif distance == "cosine":
        normalized_a = tf.nn.l2_normalize(features, axis=1)
        normalized_b = tf.nn.l2_normalize(features, axis=1)
        product = tf.matmul(normalized_a, normalized_b, adjoint_b=True)
        distance_matrix = 1.0 - product

    distance_matrix = tf.exp(-(distance_matrix / temperature))
    f = distance_matrix - tf.eye(tf.shape(features)[0])
    f = f / (stability_epsilon + tf.expand_dims(tf.reduce_sum(f, 1), 1))

    label_mask = tf.cast(
        tf.squeeze(tf.equal(labels, tf.expand_dims(labels, 1))), tf.float32
    )
    pick_probability = f * label_mask
    summed_pick_prob = tf.reduce_sum(pick_probability, 1)
    return tf.reduce_mean(-tf.math.log(stability_epsilon + summed_pick_prob))
