from setuptools import find_packages, setup


setup(
    name="g1_velocity_walk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["psutil"],
    python_requires=">=3.10",
)
