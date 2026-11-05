from setuptools import find_packages, setup


setup(
    name="mylib-cosmology",
    version="0.1.0",
    packages=find_packages(),
    description="Personal Python utilities for cosmological background calculations",
    author="Yifei Zhao",
    author_email="yfzhaoiffy@gmail.com",
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
    ],
    extras_require={
        "cobaya": ["cobaya>=3.5"],
        "test": ["pytest>=7"],
    },
)
