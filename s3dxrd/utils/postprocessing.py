import numpy as np
import vtkmodules.vtkIOXML as vtk_xml
import vtkmodules.util.numpy_support as vtk_np
import alphashape as ashape
from numpy import ndarray
from skimage import measure
from scipy import ndimage
from skimage.draw import ellipsoid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def vtk_to_numpy(vtkfile, plot=False):
    """
    Import point cloud data stored in a VTK file to a list of Numpy arrays.

    :param plot: Plot the reconstructed data as a point cloud.
    :type plot: bool
    :param vtkfile: The path of the file containing the original data as an unstructured grid.
    :type vtkfile: str
    :return: Numpy array containing the data provided in the input file.
    :rtype: tuple[list[ndarray], ndarray]
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    filereader = vtk_xml.vtkXMLUnstructuredGridReader()
    filereader.SetFileName(vtkfile)
    filereader.Update()

    data = filereader.GetOutput()
    components = ["XX", "YY", "ZZ", "YZ", "XZ", "XY"]
    coords = vtk_np.vtk_to_numpy(data.GetPoints().GetData())
    values = [vtk_np.vtk_to_numpy(data.GetPointData().GetArray(comp)) for comp in components]
    if plot:
        xcoords = [arr[0] for arr in coords]
        ycoords = [arr[1] for arr in coords]
        zcoords = [arr[2] for arr in coords]

        ax.scatter(xcoords, ycoords, zcoords)
        plt.show()

    return values, coords


def alphashape(coords):
    coords = coords.astype(int)
    coords_4d = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=int)))
    # coords_4d += (np.random.rand(np.shape(coords_4d)[0], np.shape(coords_4d)[1]) - 0.5) * 0
    transform_scale = np.array([[25, 0, 0, 0], [0, 25, 0, 0], [0, 0, 25, 0], [0, 0, 0, 1]], dtype=int)
    # transform_scale = np.eye(4, dtype=int)
    voxel_coords = np.linalg.inv(transform_scale) @ coords_4d.T
    max_vals = np.amax(voxel_coords[:3, :], axis=1)
    min_vals = np.amin(voxel_coords[:3, :], axis=1)
    shift = _min_absolute_value(max_vals, min_vals)

    transform_direction = np.array([[1, 0, 0, -shift[0]], [0, 1, 0, -shift[1]], [0, 0, 1, -shift[2]], [0, 0, 0, 1]], dtype=int)
    # transform_direction =np.eye(4, dtype=int)
    voxel_coords = transform_direction @ voxel_coords

    # nbrx = (int(max(voxel_coords[0, :])) - int(min(voxel_coords[0, :])))
    # nbry = (int(max(voxel_coords[1, :])) - int(min(voxel_coords[1, :])))
    # nbrz = (int(max(voxel_coords[2, :])) - int(min(voxel_coords[2, :])))

    voxel_volume_size = (max_vals - min_vals).astype(int)

    voxels = np.zeros((voxel_volume_size + np.ones_like(voxel_volume_size)))
    # voxels = np.zeros((600, 600, 600))
    voxel_coords = voxel_coords[:3, :].T
    for x, y, z in voxel_coords:
        voxels[int(x), int(y), int(z)] = 1

    for i in range(np.shape(voxels)[2]):
        voxels[i] = ndimage.binary_fill_holes(voxels[i]).astype(int)

    voxels = ndimage.binary_closing(voxels)
  #  voxels = ndimage.binary_fill_holes(voxels).astype(int)
    voxels = np.pad(voxels, 1)

    # coords = np.array_split(coords, np.shape(coords)[0], axis=0)
    # coords = [tuple(arr.reshape(3)) for arr in coords]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    verts, faces, normals, values = measure.marching_cubes(voxels, step_size=1)
    verts_4d = np.hstack((verts, np.ones((verts.shape[0], 1))))
    verts_coords = np.linalg.inv(transform_scale) @ (np.linalg.inv(transform_direction) @ verts_4d.T)

    xcoords = [arr[0] for arr in verts_coords.T]
    ycoords = [arr[1] for arr in verts_coords.T]
    zcoords = [arr[2] for arr in verts_coords.T]

    ax.scatter(xcoords, ycoords, zcoords)
    plt.show()

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    plt.tight_layout()
    plt.show()


def _min_absolute_value(a1, a2):
    stacked = np.vstack((a1, a2))
    indices = np.argmin(np.absolute(stacked), axis=0)
    return [int(stacked[indices[0], 0]), int(stacked[indices[1], 1]), int(stacked[indices[2], 2])]


vals, coords = vtk_to_numpy("/home/philip/Desktop/grain_stress_5.vtu")
alphashape(coords)
