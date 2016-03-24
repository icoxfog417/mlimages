import os
import random
from PIL import Image
from mlimages.util.file_api import FileAPI
import mlimages.util.log_api as LogAPI


class LabelFile():
    """
    LabelFile is constructed by the lines of (path to image file, label).
    The relative `path to image file` is allowed. If you want to do it, please set `img_root`.
    """

    def __init__(self, path, img_root="", image_label_separator=" ", debug=False):
        """
        set `img_root` if your label file uses relative path.
        """
        self.path = path
        self.img_root = img_root
        self.image_label_separator = image_label_separator

        self._debug = debug
        self._logger = LogAPI.create_logger(self.__class__.__name__, self._debug)

    def fetch(self, load_image=True):
        for im, _ in self._fetch_raw(load_image=load_image):
            yield im

    def _fetch_raw(self, load_image=True):

        with open(self.path, mode="r", encoding="utf-8") as f:
            for line in f:
                im_path, label = line.strip().split(self.image_label_separator)
                im_path = im_path if not self.img_root else os.path.join(self.img_root, im_path)
                try:
                    im = LabeledImage(im_path, label)
                    if load_image:
                        im.load()
                    yield im, line
                except Exception as ex:
                    self._logger.error(str(ex))
                    pass

    def shuffle(self, overwrite=False):
        """
        This method creates new shuffled file.
        """

        if overwrite:
            shuffled = self.path
        else:
            shuffled = FileAPI.add_ext_name(self.path, "_shuffled")

        lines = open(self.path).readlines()
        random.shuffle(lines)
        open(shuffled, "w").writelines(lines)
        self.path = shuffled


class LabeledImage():

    def __init__(self, path, label=-1):
        self.path = path # path to image file
        self.label = int(label) # label for this image
        self.image = None

    def load(self):
        self.image = Image.open(self.path)
        return self

    def to_grayscale(self):
        self.image = self.image.convert("L")
        return self

    def downscale(self, width, height=-1):
        h = height if height > 0 else width

        if self.image.size[0] >= width and self.image.size[1] >= height:
            target_v_ratio = h / width
            actual_v_ratio = self.image.size[1] / self.image.size[0]

            if target_v_ratio != actual_v_ratio:
                if actual_v_ratio > 1:
                    # height > width
                    _w = self.image.size[0]
                    self.crop_from_center(_w, int(target_v_ratio * _w))
                else:
                    # height < width
                    _h = self.image.size[1]
                    self.crop_from_center(int(target_v_ratio * _h), _h)

            self.image.thumbnail((width, h))
        else:
            self.crop_from_center(width, height)

        return self

    def crop_from_lefttop(self, left, top, width, height=-1):
        h = height if height > 0 else width
        self.image = self.image.crop((left, top, left + width, top + height))
        return self

    def crop_from_center(self, width, height=-1):
        h = height if height > 0 else width
        im_width, im_height = self.image.size
        get_bound = lambda length, size: int((length - size) / 2)
        # if image length < width, 0 padding is done
        self.crop_from_lefttop(get_bound(im_width, width), get_bound(im_height, h), width, h)
        return self

    def to_array(self, numpy_pkg, color=True):
        # https://github.com/BVLC/caffe/blob/master/python/caffe/io.py

        # image(width x height) -> matrix(column=width, row=height) = H x W matrix
        img = numpy_pkg.asarray(self.image, dtype=numpy_pkg.float32)
        if img.ndim == 2:
            # don't have color dimension
            img = img[:, :, numpy_pkg.newaxis]
            if color:
                img = numpy_pkg.tile(img, (1, 1, 3))
        elif img.shape[2] == 4:
            # RGB + A format
            img = img[:, :, :3]

        # H x W x K -> K x H x W
        img = img.transpose(2, 0, 1)

        return img

    @classmethod
    def from_array(cls, numpy_arr, label=-1):
        # K x H x W -> H x W x K
        original = numpy_arr.transpose(1, 2, 0).astype("uint8")
        image = Image.fromarray(original)
        im = LabeledImage("", label)
        im.image = image
        return im


class ImageProperty():

    def __init__(self, width, height=-1, resize_by_center=False, resize_by_downscale=False, resize_by_position=(), gray_scale=False):
        self.width = width
        self.height = height if height > 0 else self.width
        self.resize_by_center = resize_by_center
        self.resize_by_downscale = resize_by_downscale
        self.resize_by_position = resize_by_position
        self.gray_scale = gray_scale

    def convert(self, im):
        if self.gray_scale:
            im.to_grayscale()

        if len(self.resize_by_position) == 2:
            im.crop_from_lefttop(self.resize_by_position[0], self.resize_by_position[1], self.width, self.height)
        elif self.resize_by_downscale:
            im.downscale(self.width, self.height)
        else:
            im.crop_from_center(self.width, self.height)

        return im
