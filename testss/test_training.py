import os
from unittest import TestCase
from mlimages.model import ImageProperty
from mlimages.training import TrainingData
import testss.env as env


class TestLabel(TestCase):

    def test_make_mean(self):
        td = self.get_testdata()
        mean_image_file = os.path.join(os.path.dirname(td.label_file.path), "mean_image.png")

        pre_fetch = list(td.label_file.fetch())
        pre_path = td.label_file.path
        td.make_mean_image(mean_image_file)

        self.assertTrue(os.path.isfile(mean_image_file))

        generated = list(td.generate())
        self.assertEqual(len(pre_fetch), len(generated))
        self.assertNotEqual(pre_path, td.label_file.path)

        os.remove(mean_image_file)
        os.remove(td.label_file.path)

    def test_batch(self):

        # prepare
        td = self.get_testdata()
        mean_image_file = os.path.join(os.path.dirname(td.label_file.path), "mean_image.png")
        td.make_mean_image(mean_image_file)

        # make batch data
        td.shuffle()
        count = 0
        for x, y in td.generate_batches(1):
            self.assertEqual((1, 3, 32, 32), x.shape)
            self.assertEqual((1,), y.shape)
            count += 1

        self.assertEqual(env.LABEL_FILE_COUNT, count)
        os.remove(mean_image_file)
        os.remove(td.label_file.path)


    def get_testdata(self):
        p = env.get_label_file_path()
        img_root = os.path.dirname(p)
        prop = ImageProperty(32)
        td = TrainingData(p, img_root=img_root, image_property=prop)

        return td
