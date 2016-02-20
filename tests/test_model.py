import os
from unittest import TestCase
from imageset.model import API
from tests.env import get_data_folder


class TestDesign(TestCase):

    def test_prepare(self):
        p = get_data_folder()
        api = API(p)

        folders = ["one", "two"]
        relative = "/".join(folders) + "/three.txt"
        api._prepare_dir(relative)

        rmdirs = []
        for f in folders:
            p = p + "/" + f
            self.assertTrue(os.path.exists(p))
            rmdirs.append(p)

        self.assertFalse(os.path.exists(api._to_abs(relative)))

        for f in rmdirs[::-1]:
            os.rmdir(f)

    def test_download(self):
        api = API(get_data_folder())
        f = "test.xml"
        api._download("http://www.image-net.org/api/xml/structure_released.xml", f)
        self.assertTrue(api._to_abs(f))
        os.remove(api._to_abs(f))
