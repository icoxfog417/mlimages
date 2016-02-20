import os
import requests
import aiohttp
from urllib.parse import urlparse


class ImageFile():

    def __init__(self, path, label=0):
        self.path = path # path to image file
        self.label = label # label for this image


class API():

    def __init__(self, data_root, proxy="", proxy_user="", proxy_password="", limit=-1):
        self.data_root = data_root
        self.proxy = proxy
        self.proxy_user = proxy_user
        self.proxy_password = proxy_password
        self.limit = limit

    def _gather(self):
        pass

    def _set_label(self):
        pass

    def to_dataset(self):
        pass

    def create_session(self, loop):
        conn = None
        limit = self.limit
        if self.limit < 0:
            limit = 5

        if self.proxy and self.proxy_user:
            conn = aiohttp.ProxyConnector(
                loop=loop,
                limit=limit,
                proxy=self.proxy,
                proxy_auth=aiohttp.BasicAuth(self.proxy_user, self.proxy_password)
            )
        elif self.proxy:
            conn = aiohttp.ProxyConnector(loop=loop, limit=limit, proxy=self.proxy)
        else:
            conn = aiohttp.TCPConnector(loop=loop, limit=limit)

        session = aiohttp.ClientSession(connector=conn)
        return session

    def _to_abs(self, relative):
        p = os.path.join(self.data_root, "./" + relative)
        return p

    def _prepare_dir(self, relative):
        d = os.path.dirname(relative)
        if d and not os.path.exists(self._to_abs(d)):
            import re
            folders = re.split(r"\\|/", d)
            p = self.data_root
            for f in folders:
                p += os.path.sep + f
                if not os.path.exists(p):
                    os.makedirs(p)

    def _open_with_mkdir(self, relative):
        self._prepare_dir(relative)
        return open(self._to_abs(relative), "wb")

    def _get_ext(self, url):
        parsed = urlparse(url)
        root, ext = os.path.splitext(os.path.basename(parsed.path))
        if not ext:
            root, ext = os.path.splitext(os.path.basename(parsed.query.replace("=", "/")))
        ext = ext.lower()
        return ext

    def _download(self, url, relative):
        r = requests.get(url, stream=True)

        if r.ok:
            with self._open_with_mkdir(relative) as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
