import os
from unittest import TestCase
from imageset.model import FileAPI
from tests.env import get_data_folder


class TestFileAPI(TestCase):

    def test_prepare(self):
        p = get_data_folder()
        api = FileAPI(p)

        folders = ["one", "two"]
        relative = "/".join(folders) + "/three.txt"
        api.prepare_dir(relative)

        rmdirs = []
        for f in folders:
            p = p + "/" + f
            self.assertTrue(os.path.exists(p))
            rmdirs.append(p)

        self.assertFalse(os.path.exists(api.to_abs(relative)))

        for f in rmdirs[::-1]:
            os.rmdir(f)
