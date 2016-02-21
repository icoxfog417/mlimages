import os
import asyncio
import requests
import aiohttp
from urllib.parse import urlparse


class ImageFile():

    def __init__(self, path, label=0):
        self.path = path # path to image file
        self.label = label # label for this image


class API():

    def __init__(self, data_root, proxy="", proxy_user="", proxy_password="", parallel=-1, limit=-1):
        self.file_api = FileAPI(data_root)
        self.proxy = proxy
        self.proxy_user = proxy_user
        self.proxy_password = proxy_password
        self.parallel = parallel
        self.limit = limit

    def _gather(self):
        raise Exception("API has to implements gather method")

    def label_dir_auto(self, relative, fname="dataset.txt", mode="w"):
        path = self.file_api.to_abs(relative)
        self.file_api.check_file(relative, fname, mode)

        ls = os.listdir(path)
        ls.sort(key=str.lower)

        label = 0
        images = []
        for d in ls:
            rpath = self.file_api.join_relative(relative, d)
            p = self.file_api.to_abs(rpath)
            if os.path.isdir(p) and not d.startswith("."):
                labeled = self.__label_dir(rpath, label)
                self.file_api.write_iter(relative, fname, mode, labeled)
                label += 1
            elif os.path.isfile(p) and self.file_api.is_image(d):
                images.append(p)
        else:
            if len(images) > 0:
                labeled = [self.__to_line(i, label) for i in images]
                self.file_api.write_iter(relative, fname, mode, iter(labeled))

        return path

    def label_dir(self, relative, label, fname="dataset.txt", mode="w"):
        path = self.file_api.check_file(relative, fname, mode)
        labeled = self.__label_dir(relative, label)
        self.file_api.write_iter(relative, fname, mode, labeled)
        return path

    def __label_dir(self, relative, label):
        for p in self.file_api.ls_images(relative):
            yield self.__to_line(p, label)

    def __to_line(self, path, label):
        return path + " " + str(label) + "\n"

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
        except Exception as ex:
            # todo: appropriate logging
            pass

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
        sp = lambda p: [f for f in p.split("/") if f]
        ps = sp(relative) + sp(path)
        joined = "/".join(ps) + "/"
        return joined

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
                    abs = os.path.abspath(os.path.join(root, f))
                    yield abs

    def check_file(self, relative, fname, mode):
        path = self.to_abs(relative)
        fpath = os.path.join(path, fname)

        if mode == "w" and os.path.isfile(fpath):
            raise Exception("Can not remove the file")

        return os.path.abspath(fpath)

    def write_iter(self, relative, fname, mode, iterator):
        path = os.path.join(self.to_abs(relative), fname)
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
