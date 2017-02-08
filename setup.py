from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='caffe2keras',
    version='0.0.1',
    description='Converts Caffe neural nets to Keras ones',
    long_description=long_description,
    url='https://github.com/qxcv/caffe2keras',
    author='Sam Toyer',
    author_email='sam@qxcv.net',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='caffe keras converter deep learning',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['six', 'keras'],
    entry_points={
        'console_scripts': [
            'caffe2keras=caffe2keras.__main__:main',
        ]
    }
)
