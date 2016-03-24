import os
import asyncio
import concurrent.futures
import requests
import aiohttp
from mlimages.util.file_api import FileAPI
import mlimages.util.log_api as LogAPI


class API():

    def __init__(self, data_root, proxy="", proxy_user="", proxy_password="", parallel=-1, limit=-1, timeout=10, debug=False):
        self.file_api = FileAPI(data_root)
        self.proxy = proxy
        self.proxy_user = proxy_user
        self.proxy_password = proxy_password
        self.parallel = parallel if parallel > 0 else 4
        self.limit = limit
        self.timeout = timeout
        self.logger = LogAPI.create_logger(type(self).__name__, debug)

    def _gather(self):
        raise Exception("API has to implements gather method")

    def create_session(self, loop):
        conn = None

        if self.proxy and self.proxy_user:
            conn = aiohttp.ProxyConnector(
                loop=loop,
                limit=self.parallel,
                proxy=self.proxy,
                proxy_auth=aiohttp.BasicAuth(self.proxy_user, self.proxy_password)
            )
        elif self.proxy:
            conn = aiohttp.ProxyConnector(loop=loop, limit=self.parallel, proxy=self.proxy)
        else:
            conn = aiohttp.TCPConnector(loop=loop, limit=self.parallel)

        session = aiohttp.ClientSession(connector=conn)
        return session

    async def _download_images(self, session, relative, image_urls):
        self.file_api.prepare_dir(relative)
        successed = 0

        for urls in [image_urls[i:i+self.parallel] for i in range(0, len(image_urls), self.parallel)]:
            done, pendings = await asyncio.wait([self.fetch_image(session, relative, u) for u in urls])
            for d in done:
                try:
                    successed += 1 if d.result() else 0
                except:
                    pass

            if successed >= self.limit:
                break

    async def fetch_image(self, session, relative, image_url):
        fname = self.file_api.get_file_name(image_url)
        p = os.path.join(relative, fname)
        fetched = False
        try:
            with aiohttp.Timeout(self.timeout):
                async with session.get(image_url) as r:
                    if r.status == 200 and self.file_api.get_file_name(r.url) == fname:
                        c = await r.read()
                        if c:
                            with open(self.file_api.to_abs(p), "wb") as f:
                                f.write(c)
                                fetched = True
        except FileNotFoundError as ex:
            self.logger.error("{0} is not found.".format(p))
        except concurrent.futures._base.TimeoutError as tx:
            self.logger.warning("{0} is timeouted.".format(image_url))
        except Exception as ex:
            self.logger.warning("fetch image is failed. url: {0}, cause: {1}".format(image_url, str(ex)))
        return fetched

    def download_dataset(self, url, relative):
        r = requests.get(url, stream=True)

        if r.ok:
            with self.file_api.open_with_mkdir(relative) as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())

