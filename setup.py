from setuptools import setup

setup(
   name='ise',
   version='1.0.0',
   description='Ice Sheet Emulator Library',
   author='Peter Van Katwyk',
   author_email='pvnakatwyk@gmail.com',
   packages=['ise'],  #same as name
   install_requires=['pdoc'], #external packages as dependencies
)