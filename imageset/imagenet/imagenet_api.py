import asyncio
import requests
from imageset.model import API


class ImagenetAPI(API):
    NAME_URL = "http://www.image-net.org/api/text/wordnet.synset.getwords?wnid={0}"
    IMAGES_URL = "http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid={0}"
    SUBSET_URL = "http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid={0}"

    def __init__(self, data_root):
        super(ImagenetAPI, self).__init__(data_root)

    def gather(self, path, wnid):
        loop = asyncio.get_event_loop()
        session = self.create_session(loop)
        loop.run_until_complete(self.fetch_wnid(session, wnid))

    async def fetch_wnid(self, session, wnid, prefix=""):
        name_url = self.NAME_URL.format(wnid)
        images_url = self.IMAGES_URL.format(wnid)

        descs = self.__split(requests.get(name_url).text)
        name_urls = self.__split(requests.get(images_url).text)
        name_urls = [nu.split(" ") for nu in name_urls]

        folder = (descs[0] if not prefix else prefix + "/" + descs[0]) + "/"
        self._prepare_dir(folder)
        f = asyncio.wait([self.fetch(session, folder, nu) for nu in name_urls])
        return folder, await f

    async def fetch(self, session, folder, name_and_url):
        n, url = name_and_url
        ext = self._get_ext(url)
        p = "{0}{1}{2}".format(folder, n, ext)
        try:
            async with session.get(url) as r:
                if r.status == 200 and self._get_ext(r.url) == ext:
                    with open(self._to_abs(p), "wb") as f:
                        f.write(await r.read())
        except Exception as ex:
            pass

    @classmethod
    def __split(cls, text):
        ls = text.replace("\r", "").split("\n")
        return [ln.strip() for ln in ls if ln]
