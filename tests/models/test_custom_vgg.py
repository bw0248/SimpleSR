import unittest
import numpy as np
import tensorflow as tf
from simple_sr.utils.models import model_builder
from tests import _utils

INPUT_SHAPE = (None, None)


class TestCustomVgg(unittest.TestCase):

    def setUp(self):
        self.original_vgg = tf.keras.applications.vgg19.VGG19(
            input_shape=(*INPUT_SHAPE, 3), include_top=False, weights="imagenet", pooling=None
        )
        self.custom_vgg = model_builder.build_vgg_19(INPUT_SHAPE)

    def testLayerCorrespondence(self):

        for layer in self.original_vgg.layers:
            if layer.name.find("conv") > -1:
                self._assertConvLayer(layer, self.custom_vgg.get_layer(layer.name))
            elif layer.name.find("pool") > - 1:
                self._assertPoolingLayer(layer, self.custom_vgg.get_layer(layer.name))

    def testFeatureOutputCorrespondence(self):
        self.original_vgg.trainable = False
        self.custom_vgg.trainable = False
        batch1, batch2 = _utils.load_img_batches()

        for lr, hr in zip(batch1, batch2):
            lr_preprocessed = tf.keras.applications.vgg19.preprocess_input(lr)
            hr_preprocessed = tf.keras.applications.vgg19.preprocess_input(hr)

            lr_features_original = self.original_vgg(lr_preprocessed, training=False)
            lr_features_custom = self.custom_vgg(lr_preprocessed, training=False)
            np.testing.assert_array_equal(lr_features_original, lr_features_custom)

            hr_features_original = self.original_vgg(hr_preprocessed, training=False)
            hr_features_custom = self.custom_vgg(hr_preprocessed, training=False)
            np.testing.assert_array_equal(hr_features_original, hr_features_custom)

    def _assertConvLayer(self, conv_1, conv_2):
        self.assertEqual(conv_1.name, conv_2.name)
        self.assertEqual(conv_1.input_shape, conv_2.input_shape)
        self.assertEqual(conv_1.output_shape, conv_2.output_shape)
        self.assertEqual(conv_1.filters, conv_2.filters)
        self.assertEqual(conv_1.kernel_size, conv_2.kernel_size)
        self.assertEqual(conv_1.strides, conv_2.strides)
        self.assertEqual(conv_1.padding, conv_2.padding)
        np.testing.assert_array_equal(conv_1.bias.numpy(), conv_2.bias.numpy())
        self.assertEqual(conv_1.data_format, conv_2.data_format)
        self.assertEqual(conv_1.dilation_rate, conv_2.dilation_rate)
        self.assertEqual(conv_1.dtype, conv_2.dtype)
        self.assertEqual(conv_1.dynamic, conv_2.dynamic)
        self.assertEqual(conv_1.input_spec.ndim, conv_2.input_spec.ndim)
        self.assertEqual(conv_1.input_spec.axes, conv_2.input_spec.axes)
        np.testing.assert_array_equal(conv_1.kernel.numpy(), conv_2.kernel.numpy())
        self.assertEqual(conv_1.use_bias, conv_2.use_bias)

    def _assertPoolingLayer(self, pooling_1, pooling_2):
        self.assertEqual(pooling_1.name, pooling_2.name)
        self.assertEqual(pooling_1.pool_size, pooling_2.pool_size)
        self.assertEqual(pooling_1.strides, pooling_2.strides)
        self.assertEqual(pooling_1.padding, pooling_2.padding)
        self.assertEqual(pooling_1.input_shape, pooling_2.input_shape)


if __name__ == "__main__":
    unittest.main()
