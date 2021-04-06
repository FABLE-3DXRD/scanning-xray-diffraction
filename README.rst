Welcome to the s3dxrd project
===============================

This is a scientific code originally developed to adress scanning-3dxrd
strain measurements of polycrystalline materials.

Intragranular strain is computed based on a series of line integral measurements. The regression procedure involves the use of a Gaussian 
Proccess which is a statistical model that uses spatial correlation assumptions to find good fits to data. The resulting strain field is 
guaranteed to be in local static equlibrium by imposing a prior constraint on the solution function sapace.

If you want to use this code, it is strongly recomended that you have a look at the
underlying `publication`_:

    *Reconstructing intragranular strain fields in polycrystalline materials from scanning 3DXRD data, 
    Henningsson, N. A., Hall, S. A., Wright, J. P. & Hektor, J. (2020). J. Appl. Cryst. 53, 314-325.*

.. _publication: https://journals.iucr.org/j/issues/2020/02/00/nb5257/


Installation
===============================
Only manual installation is currently supported. First get the code to your local machine by:

.. code-block::

    git clone https://github.com/AxelHenningsson/scanning-xray-diffraction.git

Next go to the repository and try to install

.. code-block::

    cd scanning-xray-diffraction
    python setup build install

You will now recieve messages about dependecies that need be installed first. 
Go through these untill the build succeeds.


Documentation
===============================
Documentation is hosted seperately at `github pages`_: 

.. _github pages: https://axelhenningsson.github.io/scanning-xray-diffraction/