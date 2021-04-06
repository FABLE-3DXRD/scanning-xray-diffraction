from __future__ import print_function
import numpy as np
from skimage.transform import iradon, radon
import sys
import matplotlib.pyplot as plt


def FBP_slice( grains, flt, omegastep, rcut, ymin, ystep, number_y_scans):
    grain_masks=[]
    grain_recons=[]
    for i,g in enumerate(grains):
        sinoangles, sino, recon = FBP_grain( g, flt, \
                    ymin, ystep, omegastep, number_y_scans )
        normalised_recon = recon/recon.max()
        grain_recons.append(normalised_recon)
        mask = normalised_recon > rcut
        grain_masks.append(mask)
    update_grainshapes(grain_recons,grain_masks)
    return grain_masks


def FBP_grain( g, flt, ymin, ystep, omegastep, number_y_scans ):
    """
    Reconstruct a 2d grain shape from diffraction data using Filtered-Back-projection.
    """

    # Measured relevant data for the considered grain
    omega = flt.omega[ g.mask ].copy()
    dty = flt.dty[g.mask].copy()
    sum_intensity = flt.sum_intensity[g.mask].copy()

    if np.min(omega)<0 and np.max(omega)>0:
        # Case 1: -180 to 180 rotation and positive y-scans we make
        # a transformation back into omega=[0 180] in three steps:

        # (I) Half the intensity for peaks entering the sinogram twice.
        doublets_mask = dty<=np.abs(np.min(dty))
        sum_intensity[doublets_mask] = sum_intensity[doublets_mask]*(1/2)

        # (II) Map the negative omega values back to positive values, [0 180]
        omega_mask = omega < 0
        omega[omega_mask] = omega[omega_mask] + 180

        # (III) Flip the sign of the y-scan coordinates with negative omega values.
        dty[omega_mask]   = -dty[omega_mask]

    elif np.min(omega)>=0 and np.max(omega)<=180:
        # Case 2: 0 to 180 rotation and both negative and positive y-scans
        # nothing needs be done since this is the standrad tomography case
        pass
    else:
        raise ValueError("Scan configuration not implemented. omega=0,180 scans with positive and \
                          negative y-translations and omega=-180,180 scans with all positive      \
                          y-translations are supported.")

    # Angular range into which to bin the data (step in sinogram)
    angular_bin_size = 180./(number_y_scans)

    # Indices in sinogram for the y-scan and angles
    iy = np.round( (dty - ymin) / ystep ).astype(int)
    iom = np.round( omega / angular_bin_size ).astype(int)

    # Build the sinogram by accumulating intensity
    sinogram = np.zeros( ( number_y_scans, np.max(iom)+1 ), np.float )
    for i,I in enumerate(sum_intensity):
        dty_index   = iy[i]
        omega_index = iom[i]
        sinogram[dty_index, omega_index] += I

    # Normalise the sinogram to account for the intensity not being proportional
    # only to density but also to eta and theta and a lot of other stuff.
    normfactor = sinogram.max(axis=0)

    normfactor[normfactor==0]=1.0
    sinogram = sinogram/normfactor

    # Perform reconstruction by inverse radon transform of the sinogram
    theta = np.linspace( angular_bin_size/2., 180. - angular_bin_size/2., sinogram.shape[1] )
    back_projection = iradon( sinogram, theta=theta, output_size=number_y_scans, circle=True )


    return [], sinogram, back_projection 

def update_grainshapes( grain_recons, grain_masks):
    '''
    Update a set of grain masks based on their overlap and intensity.
    At each point the grain with strongest intensity is assigned

    Assumes that the grain recons have been normalized
    '''

    for i in range(grain_recons[0].shape[0]):
        for j in range(grain_recons[0].shape[1]):
            if conflict_exists(i,j,grain_masks):
                max_int = 0.0
                leader  = None
                for n,grain_recon in enumerate(grain_recons):
                    if grain_recon[i,j]>max_int:
                        leader = n
                        max_int = grain_recon[i,j]

                #The losers are...
                for grain_mask in grain_masks:
                    grain_mask[i,j]=0

                #And the winner is:
                grain_masks[leader][i,j]=1

def conflict_exists( i,j,grain_masks):
    '''
    Help function for update_grainshapes()
    Checks if two grain shape masks overlap at index i,j
    '''
    claimers = 0
    for grain_mask in grain_masks:
        if grain_mask[i,j]==1:
            claimers+=1
    if claimers>=2:
        return True
    else:
        return False