from setuptools import setup

setup(
   name='ise',
   version='0.0.1',
   description='Package for creating ice sheet emulators predicting future sea level rise.',
   author='Peter Van Katwyk',
   author_email='pvankatwyk@gmail.com',
   packages=['ise'],
   install_requires=['pdoc', 'numpy', 'pandas', 'scikit-learn', 'torch', 'xarray', 'tensorboard', 'matplotlib', 'seaborn', 'tqdm'],
)