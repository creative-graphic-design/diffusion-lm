from setuptools import setup, find_packages

setup(
    name="improved-diffusion",
    packages=find_packages(),
    install_requires=["blobfile", "torch", "tqdm"],
    # install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
