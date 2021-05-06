import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "utils/"))

import numpy as np
import matplotlib.pyplot as plt
from . import raw_measurements as raw_measurements
from s3dxrd.utils import topology as top 
from s3dxrd.utils import measurement_converter as mc
from s3dxrd.utils import save
import pickle
from s3dxrd.utils import slice_matcher
from xfab import tools
from scipy.ndimage.morphology import binary_opening, binary_closing

#TODO: Split this function into parts:
    # I)   Peak grain mapping + shape reconstruction.
    # II)  Cross slice mapping
    # III) Polygon representations
    # IV)  Parametric integrals
    # V)   Average strains and directions.

def peaks_to_vectors(flt_paths,
                    zpos,
                    param_path,
                    ubi_paths,
                    omegastep, 
                    ymin, 
                    ymax, 
                    ystep,
                    hkltol = 0.05,
                    nmedian = 5,
                    rcut = 0.2,
                    save_path=None ):
    
    """Convert an x-ray diffraction dataset saved in the ImageD11 format to vector format
    
    Based on an inital guess of ubi matrices all data contained in a sereis of Id11 peak files
    is analysed. Grian shapes are reconstructed and grain orientations refined. The data is converted
    into a vector format where each measurement is associated to a parametric line trhough the grain
    as well as an average approximate strain along the diffracting planes.

    The proccessed output from this function can be used to further perform regression for strain tensor
    maps using any of the in package supported regression techinques.

    Args:
        flt_paths (:obj:`list` of :obj:`string`): Absolute file paths to Id11 peaks files. These must contain
            a column named ```dty``` that stores the sample y-translations in units of microns. These translations
            should be centered at the rotation axis (dty=0 means the xray go through the rotation axis.)
        zpos (:obj:`list` of :obj:`float`): z-translations of sample corresponding to peaks in :obj:`flt_paths`.
        param_path (:obj:`string`): Absolute file path to Id11 parameters file.
        ubi_paths (:obj:`list` of :obj:`string`):  Absolute file paths to Id11 ubi matrices file.
        omegastep (:obj:`float`): Rotation intervall in degrees over wich detector images are where read out.
        ymin (:obj:`float`): Minimum y-translation of smaple in units of microns.
        ymax (:obj:`float`): Maximum y-translation of smaple in units of microns. 
        ystep (:obj:`float`): Sample y-translation stepsize.
        hkltol (:obj:`float`): Miller index tolerance for assigning peaks to a grain. Defaults to 0.05.
        nmedian (:obj:`float`): Number of median deviations of a peak to discard as outlier. Defaults to 5.
        rcut (:obj:`float`):  Relative threshold for defining grain shape from tomographic reconstrction. Defaults to 0.2.
        save_path (:obj:`string`): Path at which to save all computed for arrays. Defaults to :obj:`None`.

    Returns:
        (:obj:`dict`): dictionary with keys and values:

            **Y** (:obj:`numpy array`): Average scalar strains along scanned lines. ``shape=(N,)``
  
            **sig_m** (:obj:`numpy array`): Per measurement standard deviations. ``shape=(N,)``
  
            **entry** (:obj:`numpy array`): Entry points for line integral meaurements.  ``shape=(k,N)``
  
            **exit** (:obj:`numpy array`): Exit points for line integral meaurements.  ``shape=(k,N)``
  
            **nhat** (:obj:`numpy array`): X-ray beam direction normal in sample coordinate system. ``shape=(3,N)``
  
            **kappa** (:obj:`numpy array`): Direction of strain (Y) in sample coordinate system. ``shape=(3,N)``
  
            **L** (:obj:`numpy array`): Lengths of scanned lines. ``shape=(N,)``
  
            **nsegs** (:obj:`numpy array`): Number of segments for each line integral. ``shape=(N,)``
 
            **polygons** (:obj:`dict` of :obj:`dict` of :obj:`Shapely Polygons`): Polygonal boundary representation of grians.
  
            **polygon_orientations** (:obj:`dict` of :obj:`dict` of :obj:`numpy array`): Crystal orientation of each grain slice in polygons.
  
            **orientations** (:obj:`numpy array`): Per measurement crystal orientation (uniform along scan-line).
  
            **measurement_grain_map** (:obj:`numpy array`): dictionary keys as integers corresponding to polygons, such that
            ``polygons[str(measurement_grain_map[i])]`` gives the polygon slices dictionary of the grain
            that gave rise to measurement number i. ``shape=(N,)``

            **labeled_volume** (:obj:`numpy array`): 3d numpy array of sample with values corresponding to grain index.

            **labeled_grains** (:obj:`dict` of :obj:`dict` of :obj:`ImageD11 Grain`): Id11 grain for each grain slice in polygons.

    """
    
    print('Initiating data conversion from Id11 format to gpxrd format...')
    rm = raw_measurements.RawMeasurements(flt_paths, zpos, param_path, ubi_paths, omegastep, ymin, ymax, ystep)

    print('Mapping a total of '+str(rm.tot_nbr_peaks)+' peaks...')
    rm.map_peaks( hkltol, nmedian )

    print('Reconstructing grain topologies...')
    rm.reconstruct_grain_topology( rcut )

    # Cleanup the recons from floating pixels and holes.
    for i in range(len(rm.grain_topology_mask)):
        for j in range(len(rm.grain_topology_mask[i])):
            rm.grain_topology_mask[i][j] = binary_opening(rm.grain_topology_mask[i][j], structure=np.ones((3,3)))
            rm.grain_topology_mask[i][j] = binary_closing(rm.grain_topology_mask[i][j], structure=np.ones((3,3)))

    # Cross slice mapping of grains, giving each grain a unique label so it can be tracked across z-slices.
    print('Cross slice mapping grains...')
    labeled_volume, labeled_grains = slice_matcher.match_and_label( rm.grain_topology_mask, rm.grain_slices )

    print('Converting diffraction peaks to GP-XRD quanteties...')
    all_Y, all_sig_m, all_entry, all_exit, all_nhat, all_kappa, all_L, all_nsegs = [],[],[],[],[],[],[],[]
    orientations = None
    measurement_grain_map = []

    polygons = {}
    polygon_orientations =  {}
    for grain_indx in np.unique( labeled_volume[labeled_volume>0] ):
        polygons[str(grain_indx)] = {}
        polygon_orientations[str(grain_indx)] = {}

    for i,(zpos, peaks) in enumerate(zip( rm.zpos, rm.peak_stack)):

        print( "Working on z-position: " + str(zpos) )

        for grain_indx in np.unique( labeled_volume[:,:,i] ):

            if grain_indx==0:
                continue

            # Each grain slice is represented as a 2d polygon
            image = labeled_volume[:,:,i]==grain_indx
            grain = labeled_grains[str(grain_indx)][str(i)]
            sample_polygon = top.voxels_to_polygon( [image], pixel_size=ystep, center=(0.5,0.5) )[0]

            if 0: # DEBUG:
                top.show_polygon_and_image( sample_polygon, image, pixel_size=ystep, center=(0.5,0.5) )

            measurements = mc.convert_measurements( rm.params, grain, peaks, rm.ymin, rm.ystep, rm.omegastep )
            
            Yz     =  measurements['strain']
            kappaz =  measurements['kappa']
            sig_mz =  measurements['sig_m']
            angles =  measurements['omega']
            dty    =  measurements['dty']

            entryz, exitz, nhatz, Lz, nsegsz, bad_lines = top.get_integral_paths( angles, dty, zpos, \
                                                                    sample_polygon, show_geom=False )

            # remove line integrals that missed sample
            Yz     = np.delete(Yz, bad_lines)
            kappaz = np.delete(kappaz, bad_lines, axis=0)
            sig_mz = np.delete(sig_mz, bad_lines)

            # Build per line integral measurement orientation matrix (Nx3x3)
            U = np.zeros((1,3,3))
            U[0,:,:] = tools.ubi_to_u(grain.ubi)[:,:]
            slice_orientations = np.repeat(U, len(Yz), axis=0)
            if orientations is None:
                orientations = slice_orientations
            else:
                orientations = np.concatenate( [orientations, slice_orientations], axis=0 )

            measurement_grain_map.extend( [grain_indx]*len(Yz) )

            all_Y.extend( Yz )
            all_sig_m.extend( sig_mz )
            all_entry.append( entryz )
            all_exit.append( exitz )
            all_nhat.append( nhatz )
            all_kappa.append( kappaz )
            all_L.extend( Lz )
            all_nsegs.extend( nsegsz )

            polygons[str(grain_indx)][str(i)] = sample_polygon
            polygon_orientations[str(grain_indx)][str(i)] = U[0,:,:]

    Y, sig_m, entry, exit, nhat, kappa, L, nsegs = _repack(all_Y, all_sig_m, all_entry, all_exit, all_nhat, all_kappa, all_L, all_nsegs)
    measurement_grain_map = np.array(measurement_grain_map)

    vectors = { 'Y':Y, 
                'sig_m':sig_m, 
                'entry':entry, 
                'exit':exit, 
                'nhat':nhat, 
                'kappa':kappa, 
                'L':L, 
                'nsegs':nsegs, 
                'orientations':orientations, 
                'measurement_grain_map':measurement_grain_map, 
                'labeled_volume':labeled_volume,
                'polygons': polygons,
                'polygon_orientations': polygon_orientations }

    if save_path is not None:
        print('Writing results to disc at '+save_path+'  ...')
        save.save_object(save_path, vectors)

    return vectors


def _repack(all_Y, all_sig_m, all_entry, all_exit, all_nhat, all_kappa, all_L, all_nsegs):
    '''_repack global measurement list into numpy arrays of desired format.
    '''
    Y = np.array( all_Y )
    N = len(Y)
    
    p = np.max( [np.max(nsegs) for nsegs in all_nsegs] )

    nsegs = np.concatenate(all_nsegs).reshape( 1,N )
    sig_m  = np.array(all_sig_m).reshape( N, 1)
    L = np.concatenate(all_L).reshape( 1,N )

    entry = np.zeros( (3*p, all_entry[0].shape[1]) )
    entry[:all_entry[0].shape[0],:] = all_entry[0][:,:]
    for i in range(1,len(all_entry)):
        tmp = np.zeros( (3*p, all_entry[i].shape[1]) )
        tmp[:all_entry[i].shape[0],:] = all_entry[i][:,:]
        entry = np.concatenate( [entry,tmp], axis=1 )

    exit = np.zeros( (3*p, all_exit[0].shape[1]) )
    exit[:all_exit[0].shape[0],:] = all_exit[0][:,:]
    for i in range(1,len(all_exit)):
        tmp = np.zeros( (3*p, all_exit[i].shape[1]) )
        tmp[:all_exit[i].shape[0],:] = all_exit[i][:,:]
        exit = np.concatenate( [exit,tmp], axis=1 )

    nhat = np.concatenate( all_nhat, axis=1 )
    kappa = np.concatenate( all_kappa, axis=0 ).T
    return Y, sig_m, entry, exit, nhat, kappa, L, nsegs
