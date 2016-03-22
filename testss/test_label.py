import os
import shutil
from unittest import TestCase
from mlimages.gather.imagenet import ImagenetAPI
from mlimages.label import LabelingMachine
import testss.env as env


class TestLabel(TestCase):

    def test_label_dir(self):
        # prepare the images
        p = env.get_data_folder()
        api = ImagenetAPI(p, limit=3)
        api.gather("n01316949", include_subset=True)

        folder = "work_animal"

        # make labeled data
        machine = LabelingMachine(p)

        lf_fixed = machine.label_dir(0, path_from_root=folder, label_file=os.path.join(p, "test_label_fixed.txt"))
        lf_auto = machine.label_dir_auto(path_from_root=folder, label_file=os.path.join(p, "test_label_auto.txt"))

        f_count = 0
        for im in lf_fixed.fetch():
            self.assertEqual(0, im.label)
            f_count += 1

        label_and_path = {}
        a_count = 0
        for im in lf_auto.fetch():
            dir = os.path.dirname(im.path)
            if im.label in label_and_path:
                self.assertEqual(label_and_path[im.label], dir)
            else:
                label_and_path[im.label] = dir
            a_count += 1
        else:
            im = None # release reference

        self.assertGreater(f_count, 0)
        self.assertEquals(f_count, a_count)
        shutil.rmtree(api.file_api.to_abs(folder))
        os.remove(lf_fixed.path)
        os.remove(lf_auto.path)
