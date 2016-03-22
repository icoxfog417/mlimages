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

    def test_batch(self):
        lf = self.get_label_file()
        mean_image_file = os.path.join(os.path.dirname(lf.path), "mean_image.png")
        imp = ImageProperty(32)

        # prepare
        td = lf.to_training_data(imp)
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


    def get_label_file(self):
        p = env.get_label_file_path()
        img_root = os.path.dirname(p)
        lf = LabelFile(p, img_root=img_root)

        return lf
