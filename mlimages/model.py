import os
import asyncio
import requests
import aiohttp
from urllib.parse import urlparse
from PIL import Image


def create_logger(name, debug=False):
    logger = None
    from logging import getLogger, StreamHandler, DEBUG, INFO
    logger = getLogger(name)
    handler = StreamHandler()
    level = INFO if not debug else DEBUG
    handler.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


class ImageFile():

    def __init__(self):
        pass

    @classmethod
    def read(cls, path, img_root="", image_label_separator=None, debug=False):
        logger = create_logger(cls.__name__, debug)
        root = img_root if img_root else os.path.dirname(path)
        sep = image_label_separator
        if not sep:
            sep = lambda ln: ln.strip().split()

        with open(path, mode="r", encoding="utf-8") as f:
            for line in f:
                im_path, label = sep(line)
                im_path = im_path if not root else os.path.join(root, im_path)
                try:
                    im = IM(im_path, label)
                    im.read()
                    yield im
                except Exception as ex:
                    logger.error(str(ex))


class IM():

    def __init__(self, path, label=-1):
        self.path = path # path to image file
        self.label = label # label for this image
        self.image = None

    def read(self):
        self.image = Image.open(self.path)
        return self

    def to_grayscale(self):
        self.image = self.image.convert("L")
        return self

    def downscale(self, width, height=-1):
        h = height if height > 0 else width
        self.image.thumbnail((width, h))
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


class API():

    def __init__(self, data_root, proxy="", proxy_user="", proxy_password="", parallel=-1, limit=-1, debug=False):
        self.file_api = FileAPI(data_root)
        self.proxy = proxy
        self.proxy_user = proxy_user
        self.proxy_password = proxy_password
        self.parallel = parallel
        self.limit = limit
        self.logger = create_logger(type(self).__name__, debug)

    def _gather(self):
        raise Exception("API has to implements gather method")

    def label_dir_auto(self, relative="", label_path="", mode="w"):
        path = self.file_api.to_abs(relative)
        lpath = label_path if label_path else self.__get_default_path(relative)
        self.file_api.check_file(lpath, mode)

        ls = os.listdir(path)
        ls.sort(key=str.lower)

        label = 0
        images = []
        for d in ls:
            rpath = self.file_api.join_relative(relative, d)
            p = self.file_api.to_abs(rpath)
            if os.path.isdir(p) and not d.startswith("."):
                labeled = self.__label_dir(rpath, label)
                self.file_api.write_iter(lpath, mode, labeled)
                label += 1
            elif os.path.isfile(p) and self.file_api.is_image(d):
                images.append(self.file_api.to_rel(p))
        else:
            if len(images) > 0:
                labeled = [self.__to_line(i, label) for i in images]
                self.file_api.write_iter(lpath, mode, iter(labeled))

    def label_dir(self, label, relative="", label_path="", mode="w"):
        lpath = label_path if label_path else self.__get_default_path(relative)
        self.file_api.check_file(lpath, mode)
        labeled = self.__label_dir(relative, label)
        self.file_api.write_iter(lpath, mode, labeled)

    def __label_dir(self, relative, label):
        for p in self.file_api.ls_images(relative):
            yield self.__to_line(p, label)

    def __to_line(self, path, label):
        return path + " " + str(label) + "\n"

    def __get_default_path(self, relative=""):
        p = self.file_api.to_abs(relative)
        dirs = os.path.split(p)
        d_fname = ("dataset" if len(dirs) == 0 else dirs[-1]) + ".txt"
        d_path = os.path.abspath(os.path.join(os.getcwd() + "./" + d_fname))
        return d_path

    def create_session(self, loop):
        conn = None
        parallel = self.parallel
        if self.parallel < 0:
            parallel = 3

        if self.proxy and self.proxy_user:
            conn = aiohttp.ProxyConnector(
                loop=loop,
                limit=parallel,
                proxy=self.proxy,
                proxy_auth=aiohttp.BasicAuth(self.proxy_user, self.proxy_password)
            )
        elif self.proxy:
            conn = aiohttp.ProxyConnector(loop=loop, limit=parallel, proxy=self.proxy)
        else:
            conn = aiohttp.TCPConnector(loop=loop, limit=parallel)

        session = aiohttp.ClientSession(connector=conn)
        return session

    async def _download_images(self, session, relative, image_urls):
        self.file_api.prepare_dir(relative)
        urls = image_urls if self.limit <= 0 else image_urls[:self.limit]
        f = asyncio.wait([self.fetch_image(session, relative, u) for u in urls])
        return await f

    async def fetch_image(self, session, relative, image_url):
        fname = self.file_api.get_file_name(image_url)
        p = os.path.join(relative, fname)
        try:
            async with session.get(image_url) as r:
                if r.status == 200 and self.file_api.get_file_name(r.url) == fname:
                    with open(self.file_api.to_abs(p), "wb") as f:
                        f.write(await r.read())
        except FileNotFoundError as ex:
            self.logger.error("{0} is not found.".format(p))
        except Exception as ex:
            self.logger.warning("image is not found: {0}".format(image_url))

    def download_dataset(self, url, relative):
        r = requests.get(url, stream=True)

        if r.ok:
            with self.file_api.open_with_mkdir(relative) as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())


class FileAPI():

    def __init__(self, root):
        self.root = root

    def to_abs(self, relative):
        p = os.path.abspath(os.path.join(self.root, "./" + relative))
        return p

    def to_rel(self, abspath):
        p = os.path.relpath(abspath, self.root)
        return p

    def prepare_dir(self, relative):
        d = os.path.dirname(relative)
        if d and not os.path.exists(self.to_abs(d)):
            import re
            folders = re.split(r"\\|/", d)
            p = self.root
            for f in folders:
                p += os.path.sep + f
                if not os.path.exists(p):
                    os.makedirs(p)

    def open_with_mkdir(self, relative):
        self.prepare_dir(relative)
        return open(self.to_abs(relative), "wb")

    def get_file_name(self, url):
        parsed = urlparse(url)
        name, ext = os.path.splitext(os.path.basename(parsed.path))
        if not ext:
            name, ext = os.path.splitext(os.path.basename(parsed.query.replace("=", "/")))
        ext = ext.lower()
        fname = name + ext
        return fname

    def join_relative(self, relative, path):
        p = os.path.join(relative, path)
        relp = os.path.join(os.path.relpath(p), "")  # append final / everytime
        return relp

    def is_image(self, fname):
        extensions = [".jpeg", ".jpg", ".png", ".gif"]
        if fname.lower().endswith(tuple(extensions)):
            return True
        else:
            return False

    def ls_images(self, relative):
        path = self.to_abs(relative)
        for root, dirs, files in os.walk(path):
            for f in files:
                if self.is_image(f):
                    reldir = os.path.relpath(root, self.root)
                    rel = os.path.join(reldir, f)
                    yield rel

    def check_file(self, path, mode):
        _p = self.to_abs(path)
        if mode == "w" and os.path.isfile(_p):
            raise Exception("Can not remove the file")

    def write_iter(self, path, mode, iterator):
        _m = "wb" if mode == "w" else "ab"
        if os.path.isfile(path):
            _m = "ab"

        with open(path, mode=_m) as f:
            index = 0
            for ln in iterator:
                f.write(ln.encode("utf-8"))
                if index // 1000 == 0:
                    f.flush()
                index += 1
