# This file has to be run for one time before starting the training:
# pip install -r requirements.txt
from setuptools import setup, find_packages

setup(name='Data Challenge',
      version='0.0.0',
      description='Classification + Detection task',
      author='Ihab ASAAD',
      author_email='ihabasaad1@gmail.com',
      url='https://github.com/Ihab-Asaad/Data_Challenge',
      license='MIT',
      install_requires=open("requirements.txt", "r").read().splitlines(),
      dependency_links=['https://github.com/ildoonet/pytorch-gradual-warmup-lr/tarball/master'], 
      extras_require={
          'docs': ['sphinx', 'sphinx_rtd_theme'],
      },
      packages=find_packages(),
      keywords=[
          'Computer Vision',
          'Deep Learning',
      ])


# download the dataset and extract it to 'datasets/dataset' folder:
# link to google drive: https://drive.google.com/file/d/1H5sMjtAT_AEmjoOaElGHDN8G_v6PFcfU/view?usp=share_link

