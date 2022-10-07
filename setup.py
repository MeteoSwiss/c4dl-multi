import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="c4dlmulti",
    version="0.0.1",
    author="Jussi Leinonen",
    author_email="jussi.leinonen@meteoswiss.ch",
    description="COALITION 4 deep learning multi-hazard prediction code package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MeteoSwiss/c4dl-multihazard/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)