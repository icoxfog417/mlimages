import os
from unittest import TestCase
from mlimages.model import LabelFile, ImageProperty
import testss.env as env


class TestLabel(TestCase):

    def test_make_mean(self):
        lf = self.get_label_file()
        mean_image_file = os.path.join(os.path.dirname(lf.path), "mean_image.png")
        imp = ImageProperty(32)

        td = lf.to_training_data(imp)
        td.make_mean_image(mean_image_file)

        self.assertTrue(os.path.isfile(mean_image_file))

        lines = list(lf.fetch())
        generated = list(td.generate())
        self.assertEqual(len(lines), len(generated))

        os.remove(mean_image_file)

    def get_label_file(self):
        p = env.get_label_file_path()
        img_root = os.path.dirname(p)
        lf = LabelFile(p, img_root=img_root)

        return lf
