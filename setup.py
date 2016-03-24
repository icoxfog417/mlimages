from distutils.core import setup


setup(
    name="mlimages",
    packages=[
        "mlimages",
        "mlimages.gather",
        "mlimages.scripts",
        "mlimages.util",
    ],
    install_requires=[
        "requests",
        "aiohttp"
    ],
    version="0.4",
    description="gather image data and create training data for machine learning",
    author="icoxfog417",
    author_email="icoxfog417@yahoo.co.jp",
    url="https://github.com/icoxfog417/mlimages",
    download_url="https://github.com/icoxfog417/mlimages/tarball/0.4",
    keywords = ["imagenet", "machine learning"],
    classifiers=[],
)
