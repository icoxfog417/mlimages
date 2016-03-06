import os
import shutil
from unittest import TestCase
from mlimages.model import API
from mlimages.model import ImageFile
from mlimages.imagenet import ImagenetAPI
import tests.env as env


class TestModel(TestCase):

    def test_label_dir(self):
        p = env.get_data_folder()
        api = ImagenetAPI(p, limit=3)
        api.gather("n01316949", include_subset=True)

        folder = "work_animal"
        path = api.file_api.to_abs(folder)
        api.label_dir(0, relative=folder, label_path=os.path.join(p, "test_label_fixed.txt"))
        api.label_dir_auto(relative=folder, label_path=os.path.join(p, "test_label_auto.txt"))
        # todo have to do some check

        shutil.rmtree(path)

    def test_imread(self):
        LINES = 3
        f = env.get_imread_file()
        files = list(ImageFile.read(f))
        self.assertEqual(LINES, len(files))

        try:

            im = files[0]
            im.image.show("original image")

            im.to_grayscale()
            im.image.show("gray scaled")

            im.crop_from_center(400, 400)
            im.image.show("crop image")

            im.downscale(200)
            im.image.show("resize image")

            im.crop_from_center(220)
            im.image.show("oversize crop")

            self.assertTrue(True)

        except Exception as ex:
            raise ex

    def test_im_to_array(self):
        import numpy as np

        f = env.get_imread_file()
        files = list(ImageFile.read(f))
        im = files[0]
        im.crop_from_center(200, 300)

        arr = im.to_array(np)
        self.assertEqual(arr.shape[0], 3)
        self.assertEqual(arr.shape[1], 300)
        self.assertEqual(arr.shape[2], 200)

        arr = im.to_grayscale().to_array(np)
        self.assertEqual(arr.shape[0], 3)
        self.assertEqual(arr.shape[1], 300)
        self.assertEqual(arr.shape[2], 200)

    def test_download(self):
        api = API(env.get_data_folder())
        f = "test.xml"
        api.download_dataset("http://www.image-net.org/api/xml/structure_released.xml", f)
        self.assertTrue(api.file_api.to_abs(f))
        os.remove(api.file_api.to_abs(f))
