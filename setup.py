from setuptools import find_packages, setup

setup(
    name="ise",
    version="2.0.0",
    description="Package for creating ice sheet emulators predicting future sea level rise.",
    author="Peter Van Katwyk",
    author_email="pvankatwyk@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pdoc",
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
