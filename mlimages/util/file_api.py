import os
from urllib.parse import urlparse


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

    def write_iter(self, path, mode, iterator):
        _m = "ab" if mode in ("a", "ab") else "wb"
        with open(path, mode=_m) as f:
            index = 0
            for ln in iterator:
                f.write(ln.encode("utf-8"))
                if index // 1000 == 0:
                    f.flush()
                index += 1
