import os


def get_data_folder():
    p = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(p):
        os.mkdir(p)

    return p


def get_imread_file():
    p = os.path.join(os.path.dirname(__file__), "data_imread/label.txt")
    return p


def get_path(relative):
    return os.path.join(get_data_folder(), relative)
