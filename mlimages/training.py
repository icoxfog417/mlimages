import os
import asyncio
from PIL import Image
import numpy as np
from mlimages.model import LabelFile, LabeledImage


class TrainingData():

    def __init__(self, path, img_root="", image_label_separator=" ", image_property=None, mean_image_file="", scale=255, color=True, debug=False):
        self.label_file = LabelFile(path, img_root=img_root, image_label_separator=image_label_separator, debug=debug)
        self.image_property = image_property
        self.mean_image_file = mean_image_file
        self.scale = scale # default is 256 color(0-255)
        self.color = color

    def fetch(self):
        for im in self.label_file.fetch():
            converted = self.convert(im)
            yield  converted

    def convert(self, im):
        """
        Please override this method if you want to resize/grascale the image.
        """
        _im = im
        if self.image_property:
            _im = self.image_property.convert(im)

        return _im

    def shuffle(self):
        self.label_file.shuffle()

    def make_mean_image(self, mean_image_file=""):
        m_file = mean_image_file if mean_image_file else os.path.join(self.label_file.path, "./mean_image.png")
        im_iterator = self.label_file.fetch()

        sum_image = None
        count = 0
        for im in im_iterator:
            converted = self.convert(im)
            arr = np.asarray(converted.image)
            if sum_image is None:
                sum_image = np.ndarray(arr.shape)
                sum_image[:] = arr
            else:
                sum_image += arr
            count += 1

        mean = sum_image / count
        mean_image = Image.fromarray(mean.astype(np.uint8))
        mean_image.save(m_file)
        self.mean_image_file = m_file

    def generate(self):
        mean = self.__load_mean()

        for im in self.fetch():
            arr = self.__to_array(im, mean)
            yield  arr, im.label

    def __load_mean(self):
        mean = None
        if self.mean_image_file:
            if os.path.isfile(self.mean_image_file):
                m_image = LabeledImage(self.mean_image_file)  # mean image is already `converted` when calculation.
                m_image.load()
                mean = m_image.to_array(np, self.color)
            else:
                raise Exception("Mean image is not exist at {0}.".format(self.mean_image_file))
        else:
            self.label_file._logger.warning("Mean image is not set. So if you train the model, it will be difficult to converge.")
        return mean

    def __to_array(self, im, mean):
        arr = im.to_array(np, color=self.color)
        if mean is not None:
            arr -= mean
        if self.scale > 0:
            arr /= self.scale
        return arr

    def generate_batches(self, size):
        mean = self.__load_mean()

        async def to_array(im):
            im.load()
            converted = self.convert(im)
            return self.__to_array(converted, mean), im.label

        batch = []
        loop = asyncio.get_event_loop()

        for im in self.label_file.fetch(load_image=False):
            batch.append(to_array(im))

            if len(batch) == size:
                tasks = asyncio.wait(batch)
                done, pending = loop.run_until_complete(tasks)
                x_sample, y_sample = list(done)[0].result()
                x_batch = np.ndarray((size,) + x_sample.shape, x_sample.dtype)
                y_batch = np.ndarray((size,), np.int32)

                for j, d in enumerate(done):
                    x_batch[j], y_batch[j] = d.result()

                yield x_batch, y_batch
                i = 0
                batch.clear()

    def result_to_image(self, arr, label=-1):
        restore_arr = arr * self.scale

        if self.mean_image_file and os.path.isfile(self.mean_image_file):
            m_image = LabeledImage(self.mean_image_file)
            mean = m_image.to_array(np, self.color)
            restore_arr += mean

        im = LabeledImage.from_array(restore_arr, label)
        return im
