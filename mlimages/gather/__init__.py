import os
import asyncio
import requests
import aiohttp
from mlimages.util.file_api import FileAPI
import mlimages.util.log_api as LogAPI


class API():

    def __init__(self, data_root, proxy="", proxy_user="", proxy_password="", parallel=-1, limit=-1, debug=False):
        self.file_api = FileAPI(data_root)
        self.proxy = proxy
        self.proxy_user = proxy_user
        self.proxy_password = proxy_password
        self.parallel = parallel
        self.limit = limit
        self.logger = LogAPI.create_logger(type(self).__name__, debug)

    def _gather(self):
        raise Exception("API has to implements gather method")

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

