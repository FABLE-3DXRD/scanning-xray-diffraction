from pyevtk.hl import gridToVTK
from pyevtk.hl import pointsToVTK
import numpy as np
import pickle

def mesh2voxels(file, mesh, zpos, values, names):
    """Save arrays as fields that are paraview compatible (vtk files).

    Args:
        file (:obj:`string`): Asolute path to save the output at ending with desired filename.
        mesh (:obj:`list` of :obj:`numpy array`): Series of 2d x-y element mesh coordinate arrays.
        zpos (:obj:`list` of :obj:`float`): z-cordinates of each of the meshes in ´´´mesh´´´.
        values (:obj:`list` of :obj:`list` of :obj:`numpy array`): For each mesh this list contains
            a list for each tensor component and each mesh element. I.e values[i][j][k] gives the 
            value associated to the k:th element in the mesh[i] for the tensor component j.
        names (:obj:`list` of :obj:`string`): Names of the tensor component arrays refered to by 
            paraview. These will also be the default titles in paraview.
    """
    cellData = { name:[] for name in names }
    xg,yg,zg = [],[],[]
    for i,m in enumerate(mesh):
        for j,(x,y) in enumerate( zip( np.mean(m[:,0::2],axis=1), np.mean(m[:,1::2],axis=1) )):
            xg.append(x)
            yg.append(y)
            zg.append(zpos[i])
            for k,name in enumerate(names):
                cellData[name].append(values[i][k][j])
    for k,name in enumerate(names):
        cellData[name] = np.array(cellData[name])

    pointsToVTK(file, np.array(xg), np.array(yg), np.array(zg), cellData)

def as_vtk_voxel_volume(file, arrays, names):
    """Save arrays as fields that are paraview compatible (vtk files).

    Args:
        file (:obj:`string`): Asolute path to save the output at ending with desired filename.
        arrays (:obj:`list` of :obj:`numpy array`): The arrays to be saved.
        names (:obj:`list` of :obj:`string`): Names of the arrays refered to by paraview. These
            will also be the default titles in paraview.

    """

    x = np.arange(0, arrays[0].shape[0]+1, dtype=np.int32)
    y = np.arange(0, arrays[0].shape[1]+1, dtype=np.int32)
    z = np.arange(0, arrays[0].shape[2]+1, dtype=np.int32)

    cellData = {}
    for array,name in zip(arrays,names):
        cellData[name] = array

    gridToVTK(file, x, y, z, cellData)

def save_object( path, object_to_save ):
    with open(path, "wb") as f:
        pickle.dump(object_to_save, f, pickle.HIGHEST_PROTOCOL)

def load_vectors(path):
    """Load vectors produced and saved by s3dxrd.Id11.peaks_to_vectors().

    Args:
        path (:obj:`string`): Asolute path ending with filename of file to load.

    Returns:
        (:obj:`dict`): dictionary with fields as specified in s3dxrd.Id11.peaks_to_vectors().

    """
    with open(path, 'rb') as f:
        return pickle.load(f)
