import os
from mlimages.model import LabelFile
from mlimages.util.file_api import FileAPI


class LabelingMachine():

    def __init__(self, data_root):
        self.file_api = FileAPI(data_root)

    def label_dir_auto(self, label_file="", path_from_root="", mode="w", label_def_file=""):
        path = self.file_api.to_abs(path_from_root)
        _label_file = label_file if label_file else self.__get_default_path(path_from_root)
        ld_file = label_def_file if label_def_file else FileAPI.add_ext_name(_label_file, "_class_def")

        ls = os.listdir(path)
        ls.sort(key=str.lower)

        labels = []
        images = []
        _m = lambda: mode if len(labels) == 0 else "a"
        for d in ls:
            rpath = self.file_api.join_relative(path_from_root, d)
            p = self.file_api.to_abs(rpath)
            if os.path.isdir(p) and not d.startswith("."):
                labeled = self.__label_dir(rpath, len(labels))
                self.file_api.write_iter(_label_file, _m(), labeled)
                labels.append(rpath)
            elif os.path.isfile(p) and self.file_api.is_image(d):
                images.append(self.file_api.to_rel(p))
        else:
            if len(images) > 0:
                labeled = [self.__to_line(i, len(labels)) for i in images]
                labels.append(path_from_root if path_from_root else "ROOT")
                self.file_api.write_iter(_label_file, _m(), iter(labeled))

        if len(labels) > 0:
            self.file_api.write_iter(ld_file, "w", iter([" ".join([str(i), lb]) + "\n" for i, lb in enumerate(labels)]))
        lf = LabelFile(_label_file, img_root=self.file_api.root)
        return lf, ld_file

    def label_dir(self, label, label_file="", path_from_root="", mode="w"):
        _label_file = label_file if label_file else self.__get_default_path(path_from_root)
        labeled = self.__label_dir(path_from_root, label)
        self.file_api.write_iter(_label_file, mode, labeled)

        lf = LabelFile(_label_file, img_root=self.file_api.root)
        return lf

    def __label_dir(self, relative, label):
        for p in self.file_api.ls_images(relative):
            yield self.__to_line(p, label)

    def __to_line(self, path, label):
        return path + " " + str(label) + "\n"

    def __get_default_path(self, path_from_root=""):
        p = self.file_api.to_abs(path_from_root)
        dirs = os.path.split(p)
        d_fname = ("dataset" if len(dirs) == 0 else dirs[-1]) + ".txt"
        d_path = os.path.abspath(os.path.join(os.getcwd() + "./" + d_fname))
        return d_path

    @classmethod
    def read_label_def(cls, path):
        label_and_path = {}
        with open(path) as f:
            path_label = [ln.strip().split() for ln in f.readlines()]
            for pl in path_label:
                label_and_path[int(pl[0])] = pl[1]

        return label_and_path
