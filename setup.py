import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="s3dxrd",
    version="0.0.10",
    author="Axel Henningsson",
    author_email="nilsaxelhenningsson@gmail.com",
    description="Tools for intragranular strain estimation with s3dxrd data.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/AxelHenningsson/scanning-xray-diffraction",
    project_urls={
        "Documentation": "https://axelhenningsson.github.io/scanning-xray-diffraction/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires= [ "numpy",
                        "scipy",
                        "scikit-image",
                        "torch",
                        "h5py",
                        "matplotlib",
                        "numba",
                        "Shapely",
                        "pyevtk",
                        "xfab",
                        "fabio",
                        "ImageD11" ]
)
