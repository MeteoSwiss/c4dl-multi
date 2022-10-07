This repository contains the machine learning code used in the paper: Thunderstorm nowcasting with deep learning: a multi-hazard data fusion model, submitted to _Geophysical Research Letters_.

# Installation

You need NumPy, Scipy, Matplotlib, Seaborn, Tensorflow (2.6 used in development), Numba, Dask and NetCDF4 for Python.

Clone the repository, then, in the main directory, run
```bash
$ python setup.py develop
```
(if you plan to modify the code) or
```bash
$ python setup.py install
```
if you just want to run it.

# Downloading data

The dataset can be found at the following Zenodo repository: https://doi.org/10.5281/zenodo.6325370.

Download the NetCDF file. You can place it in the `data` directory but elsewhere on the disk works too.

# Pretrained models

The pretrained models are available at https://doi.org/10.5281/zenodo.7157986. Unzip the files `models-*.zip` found there to the `models` directory and `results.zip` to the `results` directory..

# Running

Go to the `scripts` directory and start an interactive shell. There, you can find `training.py` that contains the script you need for training and `plots_sources.py` that produces the plots from the paper.
