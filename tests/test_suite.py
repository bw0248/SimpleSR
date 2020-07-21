import unittest

from tests.data_pipeline.test_data_pipeline import TestDataPipeline
from tests.models.test_custom_vgg import TestCustomVgg
from tests.models.test_discriminator import TestDiscriminator
from tests.models.test_generator import TestGenerator
from tests.models.test_learnrate_scheduling import TestLearnrateScheduling
from tests.models.test_model_builder import TestModelBuilder
from tests.models.test_srmodel import TestSRModel
from tests.utils.image.test_image_utils import TestImageUtils
from tests.utils.image.test_metrics import TestMetrics


def test_suite():

    unit_tests = []
    suite_loader = unittest.TestLoader()
    unit_tests.append(suite_loader.loadTestsFromTestCase(TestDataPipeline))
    unit_tests.append(suite_loader.loadTestsFromTestCase(TestCustomVgg))
    unit_tests.append(suite_loader.loadTestsFromTestCase(TestDiscriminator))
    unit_tests.append(suite_loader.loadTestsFromTestCase(TestGenerator))
    unit_tests.append(suite_loader.loadTestsFromTestCase(TestLearnrateScheduling))
    unit_tests.append(suite_loader.loadTestsFromTestCase(TestModelBuilder))
    unit_tests.append(suite_loader.loadTestsFromTestCase(TestSRModel))
    unit_tests.append(suite_loader.loadTestsFromTestCase(TestImageUtils))
    unit_tests.append(suite_loader.loadTestsFromTestCase(TestMetrics))
    suite = unittest.TestSuite(unit_tests)
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(failfast=True, verbosity=3)
    runner.run(test_suite())
