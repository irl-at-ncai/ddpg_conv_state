#!/usr/bin/env python3
"""Defines the preprocessor classes for different types of inputs."""

import tensorflow as tf


class ImagePreprocessor:
    """
    The class for preprocessing image input batches.
    """
    def __init__(
            self,
            input_shape=(240, 320, 3),
            input_type=tf.float32,
            output_shape=(60, 80, 3)):
        self.input_shape = input_shape
        self.output_shape = output_shape
        # Make a tensorflow graph for processing input images
        with tf.name_scope("image_preprocessor"):
            self.input = \
                tf.placeholder(
                    name="image_input", shape=input_shape, dtype=input_type)
            self.output = \
                tf.image.resize(
                    images=self.input,
                    size=tf.constant(
                        [self.output_shape[0], self.output_shape[1]],
                        dtype=tf.int32),
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                    preserve_aspect_ratio=True)

    def process(self, sess, input_image):
        """Returns the processor input image."""
        return sess.run(self.output, {self.input: input_image})
