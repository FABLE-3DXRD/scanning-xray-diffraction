import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import shapely

"""After running diffraction_analysis.py it is time to do some regression

In this script we do a gaussian proceess regression for strain.
"""

# The meta parameters are the same still as in diffraction_analysis.py
ystep     =  5.                           # Translation stepsize in microns.
ymin      = -70.                          # Minimum sample y-translation in microns.
ymax      =  70.                          # Maximum sample y-translation in microns.
zpos      = np.array(range(1,13))*ystep   # coordinates of scanned positions in z

# We load the vector data we previously computed in diffraction_analysis.py
from s3dxrd.utils.save import load_vectors
vectors = load_vectors("./example_results/vectors.pkl")

# We must now define a compliance matrix field.
from s3dxrd.utils import compliance
H = np.load("./example_data/compliance.npy")
H_stack = np.repeat( H.reshape(1,6,6), len(vectors["Y"]), axis=0 )
H_ray = compliance.rotate_compliance( vectors["orientations"], H_stack )
H_ray  =  torch.from_numpy(H_ray).double()

# We load the converted data
from s3dxrd.utils.measurement_converter import vectors_to_torch
vectors = vectors_to_torch(vectors)

# We muct define the hyper parameters that regulates the covariance prior assumption
# This can be done by so called hyper optimisation. However, in this script we simply pick
# reasonable values as:
theta  =  torch.ones((24,)).double() * ymax/2. 

# Let's import the bayesian regression module, i.e the GP code.
from s3dxrd.regression import bayesian

# We start by defining a fourier type basis on which regression will take place 
A, B, C, D, E, F, SLambda = bayesian.defineBeltramiBasis( m=5, theta=theta, nhat=vectors["nhat"] )

print("Using a total of " + str(len(vectors["Y"])) + " measurements for regression")
print("Total number of basis functions per tensor component is: " + str(A["lambdas"].shape[1]))

# We may now build the regressor matrix for the line integral measurements
Phi = bayesian.BeltramiApproxAnisoRayMeas( A, B, C, D, E, F, H_ray, 
                                           vectors["entry"], vectors["exit"], 
                                           vectors["nsegs"], vectors["L"], 
                                           vectors["kappa"] )

# With the regression matrix precomuted regression may take place
v, r = bayesian.regression(Phi, vectors["Y"], vectors["sig_m"], SLambda)

# Before we can visualise the result as field, we need to specify what points
# we would like to plot. For this we make a grid covering the sample.
x_gv = np.arange( ymin-ystep/2, ymax+ystep/2. + ystep, ystep )      
y_gv = np.arange( ymin-ystep/2, ymax+ystep/2. + ystep, ystep )      
z_gv = np.arange( zpos[0]-ystep , zpos[-1]+ystep/2. + ystep, ystep)
[Xgrid, Ygrid, Zgrid] = np.meshgrid(x_gv, y_gv, z_gv)

# We also need to specify what compliance is applicable to each point. The   
# build_per_point_compliance_and_mask() utility function is good for this.
# Hpoint is the per point compliance over the grid of points and mask specifies 
# what points are contained by the polygonial representation of the sample.                           
Hpoint, mask = compliance.build_per_point_compliance_and_mask( H, Xgrid, Ygrid, Zgrid, 
                                                               vectors["polygons"], 
                                                               vectors["polygon_orientations"], 
                                                               zpos, ystep )

# Now we may builf the regression matrix for the point measurements
Phi_T_strain = bayesian.BeltramiApproxAnisoStrainField( A, B, C, D, E, F,
                                                        torch.from_numpy(Xgrid.ravel()),
                                                        torch.from_numpy(Ygrid.ravel()),
                                                        torch.from_numpy(Zgrid.ravel()), 
                                                        torch.from_numpy(Hpoint) )

# And finally we can predict the mean value and uncertainty at the grid locations
pred, chol_var = bayesian.predict(Phi_T_strain, v, r)

# To get the standard deviations from the childesky factor of the variances we do
stds = torch.norm( chol_var, dim=0 )

# Now it is only a mater of unpacking the results as XX,YY etc components of strain
strain_stds = [stds[i::6].reshape(len(x_gv),len(y_gv),len(z_gv) ).data.detach().numpy()*mask for i in range(6)]
strains = [pred[i::6].reshape(len(x_gv),len(y_gv),len(z_gv) ).data.detach().numpy()*mask for i in range(6)]

# finally we save the 3d fields as voxel volumes to visualise in paraview
from s3dxrd.utils.save import as_vtk_voxel_volume
titles = ["XX","YY","ZZ","XY","XZ","YZ"]
as_vtk_voxel_volume("./example_results/gp_strain", strains, titles)
as_vtk_voxel_volume("./example_results/gp_strain_stds", strain_stds, titles)