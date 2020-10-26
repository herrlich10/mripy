import setuptools
import mripy
from os import path

setup_dir = path.dirname(path.realpath(__file__))
with open(f'{setup_dir}/README.rst', 'r') as f:
    long_description = f.read()

setuptools.setup(
   name='mripy',
   version=mripy.__version__,
   author='herrlich10',
   author_email='herrlich10@gmail.com',
   description='High resolution fMRI data analysis in Python, based on AFNI, FreeSurfer, ANTs, and many other tools.',
   long_description=long_description,
   long_description_content_type='text/x-rst',
   url='https://github.com/herrlich10/mripy',
   packages=setuptools.find_packages(),
   include_package_data=True,
   install_requires=[
       # 'numpy',
   ],
   classifiers=(
       'Programming Language :: Python :: 3',
       'Operating System :: OS Independent',
       'License :: OSI Approved :: MIT License',
       'Intended Audience :: Developers',
       'Intended Audience :: Science/Research',
       'Topic :: Scientific/Engineering :: Information Analysis',
       'Topic :: Software Development :: Libraries',
       'Topic :: Utilities',
   ),
   python_requires='>=3.6',
)
