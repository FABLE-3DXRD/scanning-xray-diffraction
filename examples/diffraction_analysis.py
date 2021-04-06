import numpy as np
import matplotlib.pyplot as plt

"""Let us convert some ID11 ESRF peak files to vector format using the s3dxrd package.

This is the first step before any regression can be performed.
"""

# We will work with a simulated data set that has the following meta parameters
ystep     =  5.                           # Translation stepsize in microns.
ymin      = -70.                          # Minimum sample y-translation in microns.
ymax      =  70.                          # Maximum sample y-translation in microns.
zpos      = np.array(range(1,13))*ystep   # coordinates of scanned positions in z
omegastep = 1.0                           # Detector readout integration interval in degrees
hkltol    = 0.05                          # Miller index tolerance for mapping peaks.
nmedian   = np.inf                        # discard outliers, np.inf keeps all peaks
rcut      = 0.35                          # Relative segmentation threshold for grain density map.

# We build the paths to the example data peak files and ubi files and parameters file
ubi_paths  = ["./example_data/example_grain_"+"z"+str(i)+".ubi" for i in range(len(zpos))]
param_path =  "./example_data/example_grain_z3.par"
flt_paths  = ["./example_data/example_grain_"+"z"+str(i)+".flt" for i in range(len(zpos))]

# We import the Id11 module to convert the peak files to vectors.
from s3dxrd.measurement.Id11 import peaks_to_vectors
vectors = peaks_to_vectors( flt_paths, zpos, param_path,
                            ubi_paths, omegastep, ymin,
                            ymax, ystep, hkltol, nmedian,
                            rcut, save_path = "./example_results/vectors.pkl" )

# Vectors is now a dictionary with keys documented in s3dxrd.measurement.Id11.peaks_to_vectors()
for key in vectors: print(key, " : ", type(vectors[key]))

# Now, Let's see what the reconstructed grains look like in 3d
labeled_volume = vectors["labeled_volume"]

colors = np.zeros(labeled_volume.shape+(3,))
for i in np.unique(labeled_volume[labeled_volume!=0]):
    colors[labeled_volume==i, :] = np.random.rand(3,)

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(labeled_volume!=0, facecolors=colors, edgecolor='k')
plt.show()