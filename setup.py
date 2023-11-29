from setuptools import setup, find_packages

setup(
    name='gaze3d',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        # any other dependencies your module needs
    ],
)