Welcome to the s3dxrd project
===============================

This is a scientific code originally developed to adress scanning-3dxrd
strain measurements of polycrystalline materials.

Intragranular strain is computed based on a series of line integral measurements. The regression procedure involves the use of a Gaussian 
Proccess which is a statistical model that uses spatial correlation assumptions to find good fits to data. The resulting strain field is 
guaranteed to be in local static equlibrium by imposing a prior constraint on the solution function sapace.

If you want to use this code, it is strongly recomended that you have a look at the
underlying `publication`_:

    *Intragranular Strain Estimation in Far-Field Scanning X-ray 
    Diffraction using a Gaussian Processes, Axel Henningsson and Johannes Hendriks, 
    2021, under review for Journal of Applied Crystalography*

.. _publication: https://www.researchgate.net/publication/349520623_Intragranular_Strain_Estimation_in_Far-Field_Scanning_X-ray_Diffraction_using_a_Gaussian_Processes


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