Welcome to the s3dxrd package
===============================

This is a scientific code originally developed to adress scanning-3d-xray-diffraction (s3dxrd)
strain measurements of polycrystalline materials.

Intragranular strain is computed based on a series of line integral measurements. The s3dxrd package supports
regression either by a simple weighted least squares approach or alternatively by a Gaussian Proccess. The later
statistical model uses spatial correlation assumptions and an equlibrium prior to find good fits to data.

If you want to use this code, it is strongly recomended that you have a look at `the underlying publication`_: 
describing the weighted least squares approach (named "ASR" in the paper)

    *Reconstructing intragranular strain fields in polycrystalline materials from scanning 3DXRD data, 
    Henningsson, N. A., Hall, S. A., Wright, J. P. & Hektor, J. (2020). J. Appl. Cryst. 53, 314-325.*

.. _the underlying publication: https://journals.iucr.org/j/issues/2020/02/00/nb5257/

A preprint describing the Gaussian Process regression procedure is also available `here`_:

    *Intragranular Strain Estimation in Far-Field Scanning X-ray Diffraction using a Gaussian Processes, 
    Axel Henningsson and Johannes Hendriks. (2021). arXiv Preprint.*

.. _here: https://arxiv.org/abs/2102.11018

This paper may also help the user to understand some of the mathematical notation hinted at in the code.

Installation
===============================
Installation via pip is technically possible as

.. code-block::

    pip3 install s3dxrd

However, the latest ImageD11 1.9.7 is not currently available at pypi, thus this dependecy
must be manually installed first. `Checkout the repo for how to do this`_:

.. _Checkout the repo for how to do this: https://github.com/FABLE-3DXRD/ImageD11

For manuall installation, first get the code to your local machine by:

.. code-block::

    git clone https://github.com/AxelHenningsson/scanning-xray-diffraction.git

Next go to the repository and try to install

.. code-block::

    cd scanning-xray-diffraction
    python setup.py build install

You will now recieve messages about dependecies that need be installed first. 
Go through these untill the build succeeds.


Documentation
===============================
Documentation is hosted seperately at `github pages`_: 

.. _github pages: https://axelhenningsson.github.io/scanning-xray-diffraction/
