import unittest
import tensorflow as tf
from simple_sr.utils.image import metrics
from tests import _utils


class TestMetrics(unittest.TestCase):
    batch1_normalized, batch2_normalized = _utils.load_img_batches(normalize=True)
    batch1, batch2 = _utils.load_img_batches(normalize=False)

    def testPSNR(self):
        for b1, b2 in zip(self.batch1, self.batch2):
            self.assertEqual(float("inf"), metrics.psnr(b1[0], b1[0], max_val=255).numpy())
            self.assertEqual(float("inf"), metrics.psnr(b2[0], b2[0], max_val=255).numpy())
            self.assertNotEqual(float("inf"), metrics.psnr(b1[0], b2[0], max_val=255).numpy())
            batch_psnr = metrics.psnr(b1, b2, max_val=255).numpy()
            self.assertEqual(b1.shape[0], len(batch_psnr))
            for _psnr in batch_psnr:
                self.assertNotEqual(float("inf"), _psnr)

        for b1, b2 in zip(self.batch1_normalized, self.batch2_normalized):
            self.assertEqual(float("inf"), metrics.psnr(b1[0], b1[0], max_val=1.0))
            self.assertNotEqual(float("inf"), metrics.psnr(b2[0], b2[1], max_val=1.0))
            batch_psnr = metrics.psnr(b1, b2, max_val=1.0).numpy()
            self.assertEqual(b1.shape[0], len(batch_psnr))
            for _psnr in batch_psnr:
                self.assertNotEqual(float("inf"), _psnr)

    def testPSNROnY(self):
        for b1, b2 in zip(self.batch1_normalized, self.batch2_normalized):
            self.assertEqual(float("inf"), metrics.psnr_on_y(b1[0], b1[0]))
            self.assertNotEqual(float("inf"), metrics.psnr_on_y(b1[0], b2[0]))
            self.assertNotEqual(float("inf"), metrics.psnr_on_y(b1, b2).numpy().any())

            t1_y = tf.image.rgb_to_yuv(b1[0])[:, :, 0]
            t2_y = tf.image.rgb_to_yuv(b2[0])[:, :, 0]
            psnr_y_ref = -10 * tf.math.log(
                tf.math.reduce_mean(tf.math.squared_difference(t1_y, t2_y))
            ) / tf.math.log(10.0)
            psnr_y = metrics.psnr_on_y(b1[0], b2[0], max_val=1.0)
            self.assertAlmostEqual(psnr_y_ref, psnr_y.numpy(), delta=1e-5)

    def testSSIM(self):
        for b1, b2 in zip(self.batch1, self.batch2):
            self.assertEqual(1.0, metrics.ssim(b1[0], b1[0], max_val=255).numpy())
            self.assertEqual(1.0, metrics.ssim(b2[0], b2[0], max_val=255).numpy())
            self.assertNotEqual(1.0, metrics.ssim(b1[0], b2[0], max_val=255).numpy())
            batch_ssim = metrics.ssim(b1, b2, max_val=255).numpy()
            self.assertEqual(b1.shape[0], len(batch_ssim))
            for _ssim in batch_ssim:
                self.assertNotEqual(float("inf"), _ssim)

        for b1, b2 in zip(self.batch1_normalized, self.batch2_normalized):
            self.assertEqual(1.0, metrics.ssim(b1[0], b1[0], max_val=1.0))
            self.assertNotEqual(1.0, metrics.ssim(b2[0], b2[1], max_val=1.0))
            batch_ssim = metrics.ssim(b1, b2, max_val=1.0).numpy()
            self.assertEqual(b1.shape[0], len(batch_ssim))
            for _ssim in batch_ssim:
                self.assertNotEqual(float("inf"), _ssim)


if __name__ == "__main__":
    unittest.main()
