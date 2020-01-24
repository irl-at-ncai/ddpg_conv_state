#!/usr/bin/env python3
"""Defines the preprocessor classes for different types of inputs."""

import tensorflow as tf
import numpy as np


class StateConcatenator:
    """
    The class for preprocessing robot state inputs into desired outputs.
    """
    def __init__(self, output_name):
        self.output_name = output_name

    # pylint: disable=unused-argument
    def process(self, inputs, sess=None):
        """
        Redefines the input dictionary according to required reshaping of
        input.
        """
        return np.concatenate(inputs)


class ImagePreprocessor:
    """
    The class for preprocessing image input batches.
    """
    def __init__(
            self,
            input_shape=(240, 320, 4),
            input_type=tf.float32,
            output_shape=(60, 80, 4)):
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

    def process(self, input_image, sess):
        """Returns the processor input image."""
        return sess.run(self.output, {self.input: input_image})
