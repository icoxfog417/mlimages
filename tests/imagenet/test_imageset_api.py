import os
import shutil
from unittest import TestCase
from imageset.imagenet import ImagenetAPI
import tests.env as env


class TestDesign(TestCase):

    def test_prepare(self):
        p = env.get_data_folder()
        api = ImagenetAPI(p)
        api.gather(p, "n11531193")

        target = env.get_path("wilding")
        self.assertTrue(os.path.isdir(target))
        #shutil.rmtree(target)
