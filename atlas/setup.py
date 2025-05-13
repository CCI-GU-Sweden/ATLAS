from setuptools import setup, find_packages
from pathlib import Path

# Get the directory containing setup.py
this_directory = Path(__file__).parent

# Read README.md from one directory up
long_description = (this_directory.parent / "README.md").read_text(encoding="utf-8")



setup(
    name="atlas",
    version="0.1.0",
    description="Your package description here",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-username/ATLAS",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        # List your dependencies here, e.g.:
        # "numpy>=1.19.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)