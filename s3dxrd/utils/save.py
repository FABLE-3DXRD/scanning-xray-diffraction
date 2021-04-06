from pyevtk.hl import gridToVTK
import numpy as np
import pickle

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
