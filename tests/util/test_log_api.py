import os
import shutil
from unittest import TestCase
from mlimages.util.log_api import create_file_logger
from testss.env import get_data_folder


class TestLogAPI(TestCase):

    def test_file_log(self):
        p = get_data_folder()
        log_file = "log.txt"
        logger, root = create_file_logger(p, "test_logger", file_name=log_file)

        logger.info("test1")
        logger.info("test2")

        lines = open(os.path.join(root, log_file)).readlines()
        self.assertEqual(len(lines), 2)
        logger.close()
        shutil.rmtree(root)
