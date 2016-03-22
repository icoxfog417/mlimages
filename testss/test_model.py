import os
import shutil
from unittest import TestCase
from mlimages.model import LabeledImage, LabelFile
import numpy as np
import testss.env as env


class TestModel(TestCase):

    def test_imread(self):

        try:

            im = self.get_labeled_image()
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

        im = self.get_labeled_image()
        im.crop_from_center(200, 300)

        arr = im.to_array(np)
        self.assertEqual(arr.shape[0], 3)
        self.assertEqual(arr.shape[1], 300)
        self.assertEqual(arr.shape[2], 200)

        arr = im.to_grayscale().to_array(np)
        self.assertEqual(arr.shape[0], 3)
        self.assertEqual(arr.shape[1], 300)
        self.assertEqual(arr.shape[2], 200)

    def test_to_array_from_array(self):
        im = self.get_labeled_image()

        arr = im.to_array(np)
        restored = LabeledImage.from_array(arr)
        restored_arr = restored.to_array(np)

        self.assertEqual(arr.shape, restored_arr.shape)
        self.assertEqual(arr.sum(), restored_arr.sum())
        # restored.image.show("restored image")

    def test_label_file(self):
        LINES = 3
        p = env.get_label_file_path()
        img_root = os.path.dirname(p)

        lf = LabelFile(p, img_root=img_root)
        files = list(lf.fetch())
        self.assertEqual(LINES, len(files))

    def get_labeled_image(self):
        p = env.get_image_path()
        im = LabeledImage(p, 1)
        im.load()
        return im
