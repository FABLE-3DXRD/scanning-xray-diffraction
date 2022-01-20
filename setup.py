import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="s3dxrd",
    version="0.0.8",
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
    install_requires= [ "numpy>=1.20.0",
                        "scipy",
                        "scikit-image>=0.17.2",
                        "torch>=1.6.0",
                        "h5py",
                        "matplotlib",
                        "numba>=0.53.1",
                        "Shapely>=1.7.0",
                        "pyevtk>=1.1.2",
                        "xfab>=0.0.4",
                        "fabio>=0.10.2",
                        "ImageD11>=1.9.7" ]
)
