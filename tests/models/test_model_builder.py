import unittest
import tensorflow as tf
from simple_sr.utils.models import model_builder

INPUT_DIMS = (128, 128)
SCALING_FACTORS = [2, 4, 8]


class TestModelBuilder(unittest.TestCase):

    def test_build_srresnet(self):
        model = model_builder.build_resnet(
            upsample_factor=2, num_filters=64, num_res_blocks=16, momentum=0.8,
            input_dims=(None, None), batch_normalization=True
        )
        self._assert_input_output(model, (None, None, None, 3), (None, None, None, 3))

        for scale in SCALING_FACTORS:
            m = model_builder.build_resnet(
                upsample_factor=scale, input_dims=INPUT_DIMS
            )
            self._assert_input_output(
                m, (None, *INPUT_DIMS, 3), (None, scale * INPUT_DIMS[0], scale * INPUT_DIMS[1], 3)
            )

    def test_build_rrdb(self):
        model = model_builder.build_enhanced_resnet(
            upsample_factor=2, input_dims=(None, None)
        )
        self._assert_input_output(model, (None, None, None, 3), (None, None, None, 3))

        for scale in SCALING_FACTORS:
            m = model_builder.build_enhanced_resnet(
                upsample_factor=scale, input_dims=INPUT_DIMS
            )
            self._assert_input_output(
                m, (None, *INPUT_DIMS, 3), (None, scale * INPUT_DIMS[0], scale * INPUT_DIMS[1], 3)
            )

    def test_build_discriminator(self):
        std_disc = model_builder.build_discriminator(input_dims=(128, 128), relativistic=False)
        self._assert_input_output(std_disc, (None, *INPUT_DIMS, 3), (None, 1))
        self.assertEqual(1, len(std_disc.outputs))
        self.assertTrue("activation" in std_disc.output_names[0])

        relativistic = model_builder.build_discriminator(input_dims=(128, 128), relativistic=True)
        self._assert_input_output(relativistic, (None, *INPUT_DIMS, 3), (None, 1))
        self.assertEqual(1, len(relativistic.outputs))
        self.assertTrue("dense" in relativistic.output_names[0])

    def _assert_input_output(self, model, expected_input_dims, expected_output_dims):
        self.assertTrue(model.built)
        self.assertEqual(expected_input_dims, model.input_shape)
        self.assertEqual(expected_output_dims, model.output_shape)


if __name__ == '__main__':
    unittest.main()
