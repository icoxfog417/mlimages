from distutils.core import setup


setup(
    name="mlimages",
    packages=[
        "mlimages",
        "mlimages.imagenet"
    ],
    install_requires=[
        "requests",
        "aiohttp"
    ],
    version="0.1",
    description="gather and create image dataset for machine learning",
    author="icoxfog417",
    author_email="icoxfog417@yahoo.co.jp",
    url="https://github.com/icoxfog417/mlimages",
    download_url="https://github.com/icoxfog417/mlimages/tarball/0.1",
    keywords = ["imagenet", "machine learning"],
    classifiers=[],
)
