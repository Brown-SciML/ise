from setuptools import find_packages, setup

setup(
    name="ise",
    version="1.0.1",
    description="Package for creating ice sheet emulators predicting future sea level rise.",
    author="Peter Van Katwyk",
    author_email="pvankatwyk@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "xarray",
        "tensorboard",
        "matplotlib",
        "seaborn",
        "tqdm",
        "cftime",
        "netcdf4",
        "nflows",
    ],
)
