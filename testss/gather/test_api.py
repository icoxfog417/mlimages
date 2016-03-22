import os
import shutil
from unittest import TestCase
from mlimages.gather import API
import testss.env as env


class TestAPI(TestCase):

    def test_download(self):
        api = API(env.get_data_folder())
        f = "test.md"
        api.download_dataset("https://raw.githubusercontent.com/icoxfog417/mlimages/master/README.md", f)
        self.assertTrue(api.file_api.to_abs(f))
        os.remove(api.file_api.to_abs(f))
