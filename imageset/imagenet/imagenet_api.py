import asyncio
import requests
from imageset.model import API


class ImagenetAPI(API):
    NAME_URL = "http://www.image-net.org/api/text/wordnet.synset.getwords?wnid={0}"
    IMAGES_URL = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={0}"
    SUBSET_URL = "http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid={0}"

    def gather(self, wnid, relative="", include_subset=False):
        loop = asyncio.get_event_loop()
        session = self.create_session(loop)
        folders = []

        f = loop.run_until_complete(self.download_images(session, wnid, relative))
        folders.append(f)

        if include_subset:
            wnids = self._get_subsets(wnid)
            path = self.file_api.join_relative(relative, f)
            downloads = asyncio.wait([self.download_images(session, wnid, path) for wnid in wnids])
            fs = loop.run_until_complete(downloads)
            folders += [f.result() for f in fs[0]]

        session.close()

        return folders

    def _get_subsets(self, wnid):
        subset_url = self.SUBSET_URL.format(wnid)
        wnids = self.__split(requests.get(subset_url).text)
        wnids = [w[1:] for w in wnids[1:]]
        return wnids

    async def download_images(self, session, wnid, relative=""):
        name_url = self.NAME_URL.format(wnid)
        images_url = self.IMAGES_URL.format(wnid)

        descs = self.__split(requests.get(name_url).text)
        urls = self.__split(requests.get(images_url).text)

        folder = self.file_api.join_relative(relative, descs[0].lower().replace(" ", "_"))
        await self._download_images(session, folder, urls)
        return folder

    @classmethod
    def __split(cls, text):
        ls = text.replace("\r", "").split("\n")
        return [ln.strip() for ln in ls if ln]
