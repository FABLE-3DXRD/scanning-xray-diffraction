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

The paper describing the Gaussian Process regression procedure is also available `here`_:

    *Intragranular Strain Estimation in Far-Field Scanning X-ray Diffraction using a Gaussian Processes, 
    Henningsson, A. & Hendriks, J. (2021). J. Appl. Cryst. 54.*

.. _here: https://journals.iucr.org/j/issues/2021/04/00/nb5298/

This paper may also help the user to understand some of the mathematical notation hinted at in the code.

Installation
===============================
Installation via pip is available as

.. code-block::

    pip3 install s3dxrd

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
