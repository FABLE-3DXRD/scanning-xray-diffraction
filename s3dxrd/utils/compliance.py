import numpy as np
import shapely

def rotate_compliance( U, H ):
    """Rotate a series of 6x6 "tensor matrices" according to a series of 3x3 rotation matrices.
    
    Args:
        U (:obj:`numpy array`): Orientation/rotation tensors, ```shape=(Nx3x3)``` 
        H (:obj:`numpy array`): Compliance matrices, ```shape=(Nx6x6)```

    Returns:
        (:obj:`numpy array`) of ```shape=(Nx6x6)``` rotated compliance matrices.

    """
    R = _get_rotation_matrix( U )
    H_rot = np.zeros( H.shape )
    for i in range(H.shape[0]):
        H_rot[i,:,:] = (R[i,:,:].dot(H[i,:,:])).dot( np.linalg.inv(R[i,:,:]) )
    return H_rot

def _get_rotation_matrix( U ):
    """Return the rotation matrix, R, that corresponds to rotating a symmetric 3x3 tensor eps by U as U*eps*U.T.
    
    R*eps_bar contains the same values in vector format where eps_bar is 6x1 stack of eps uniqie values as

    Args:
        U (:obj:`numpy array`): Orientation/rotation tensors, ```shape=(Nx3x3)``` 

    Returns:
        (:obj:`numpy array`) of ```shape=(Nx6x6)``` rotation matrix.

    """
    # Format verified by sympy (see rotations.py in docs)
    R = np.array([[U[:,0,0]**2, U[:,0,1]**2, U[:,0,2]**2, 2*U[:,0,0]*U[:,0,1], 2*U[:,0,0]*U[:,0,2], 2*U[:,0,1]*U[:,0,2]], 
        [U[:,1,0]**2, U[:,1,1]**2, U[:,1,2]**2, 2*U[:,1,0]*U[:,1,1], 2*U[:,1,0]*U[:,1,2], 2*U[:,1,1]*U[:,1,2]], 
        [U[:,2,0]**2, U[:,2,1]**2, U[:,2,2]**2, 2*U[:,2,0]*U[:,2,1], 2*U[:,2,0]*U[:,2,2], 2*U[:,2,1]*U[:,2,2]], 
        [U[:,0,0]*U[:,1,0], U[:,0,1]*U[:,1,1], U[:,0,2]*U[:,1,2], U[:,0,0]*U[:,1,1] + U[:,0,1]*U[:,1,0], U[:,0,0]*U[:,1,2] + U[:,0,2]*U[:,1,0], U[:,0,1]*U[:,1,2] + U[:,0,2]*U[:,1,1]], 
        [U[:,0,0]*U[:,2,0], U[:,0,1]*U[:,2,1], U[:,0,2]*U[:,2,2], U[:,0,0]*U[:,2,1] + U[:,0,1]*U[:,2,0], U[:,0,0]*U[:,2,2] + U[:,0,2]*U[:,2,0], U[:,0,1]*U[:,2,2] + U[:,0,2]*U[:,2,1]], 
        [U[:,1,0]*U[:,2,0], U[:,1,1]*U[:,2,1], U[:,1,2]*U[:,2,2], U[:,1,0]*U[:,2,1] + U[:,1,1]*U[:,2,0], U[:,1,0]*U[:,2,2] + U[:,1,2]*U[:,2,0], U[:,1,1]*U[:,2,2] + U[:,1,2]*U[:,2,1]]])
    R = np.moveaxis(R, 0, 2)
    R = np.moveaxis(R, 0, 2)
    return R

def build_per_point_compliance_and_mask( compliance, Xgrid, Ygrid, Zgrid, polygons, polygon_orientations, zpos, ystep ):
    """Given a material compliance and orientation field, build a per point complance field and density mask.

    This function iterates over ```polygons``` to check if the points in ```Xgrid```, ```Ygrid```, ```Zgrid```
    are contained by any polygon. If so, the compliance of the given polygon is assigned to the point, as specified
    by polygon_orientations and compliance.

    Args:
        compliance (:obj:`numpy array`): Compliance matrix, ``shape=(6,6)``.
        Xgrid, Ygrid, Zgrid (:obj:`numpy array`): x,y,z coordinate arrays, ``shape=(m,n,l)``.
        polygons (:obj:`dict`): See s3dxrd.measurements.Id11.peaks_to_vectors()
        polygon_orientations (:obj:`dict`): See s3dxrd.measurements.Id11.peaks_to_vectors()
        zpos (:obj:`list`): z-coordinates of polygons.
        ystep (:obj:`float`): Distance in microns between gridpoints.

    Returns:
        (:obj:`tuple`): tuple with values:

            **Hpoint** (:obj:`numpy array`): Per gridpoint compliance of ``shape=(m,n,l,6,6)``. Zero outside of mask.
  
            **mask** (:obj:`numpy array`):  Density mask of ``shape=(m,n,l)``. np.nan is assigned to empty points.
  
    """
    Hpoint = np.zeros(  (Xgrid.shape[0],Xgrid.shape[1],Xgrid.shape[2], 6, 6) )
    mask = np.full( Xgrid.shape, np.nan )
    grainH  = {}
    for key1 in polygon_orientations:
        grainH[key1] = {}
        for key2 in polygon_orientations[key1]:
            orient = polygon_orientations[key1][key2]
            grainH[key1][key2] = rotate_compliance( orient.reshape(1,3,3), compliance.reshape(1,6,6 ) )
    for k in range(Xgrid.shape[2]):
        polygon_z_slice_indx = str(np.argmin( np.abs( zpos - Zgrid[0,0,k] ) ))
        for i in range(Xgrid.shape[0]):
            for j in range(Xgrid.shape[1]):
                p = shapely.geometry.Point(Xgrid[i,j,k],Ygrid[i,j,k])
                for key in polygons:
                    if polygon_z_slice_indx in polygons[key]:
                        if polygons[key][polygon_z_slice_indx].buffer( ystep/4. ).contains( p ):
                            Hpoint[i,j,k,:,:] = grainH[key][polygon_z_slice_indx]
                            if np.min(np.abs( zpos - Zgrid[i,j,k] ))<=ystep/2.:
                                mask[i,j,k] = 1
                            break
    Hpoint = Hpoint.reshape(Hpoint.shape[0]*Hpoint.shape[1]*Hpoint.shape[2],6,6)
    return Hpoint, mask