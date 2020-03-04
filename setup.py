import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


requirements = [
    'numpy',
    'torch',
    'torchvision',
    'Pillow',
]


setup(
    name="franks_ml_model",
    version="0.0.1",
    url="https://github.com/ml-illustrated/mlillustrated_franks_ml_model",
    author="Frank Torch",
    author_email="frank@my-co.com",
    description="Example Model for classifying images",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6"
    ],
)
