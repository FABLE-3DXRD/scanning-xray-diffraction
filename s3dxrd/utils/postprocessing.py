import datetime
import numpy as np
import scipy.ndimage
import vtkmodules.vtkIOXML as vtk_xml
import vtkmodules.util.numpy_support as vtk_np
from numpy import ndarray
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numba import jit


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


def alphashape(coords, nlayers=1, plot=False):
    # TODO: Implement multi-layer alpha shape calculation.
    """
    Calculate the alpha shape (the concave hull) for a point cloud consisting of a given set of
    three-dimensional coordinates. The code presumes that the coordinates are given in microns and that the
    measurements are taken 25 microns apart.

    :param coords: List of coordinates in x, y and z for the different points of the point cloud.
    :type coords:  ndarray
    :param nlayers: Number of layers in the alpha shape, Defaults to one (the outermost layer).
    :type nlayers: int
    :param plot: Toggle plotting of the alpha shape as a point cloud and as a tessellated mesh body. Defaults to False.
    :type plot: bool
    :return: The coordinates of the points in the point cloud corresponding to the alpha shape.
    :rtype: ndarray
    """
    coords_4d = np.hstack((coords, np.ones((coords.shape[0], 1))))

    transform_scale = np.array([[25., 0, 0, 0], [0, 25., 0, 0], [0, 0, 25., 0], [0, 0, 0, 1.]])
    inv_transform_scale = np.array([[1 / 25., 0, 0, 0], [0, 1 / 25., 0, 0], [0, 0, 1 / 25., 0], [0, 0, 0, 1.]])
    voxel_coords = (inv_transform_scale @ coords_4d.T)

    max_vals = np.amax(voxel_coords[:3, :], axis=1)
    min_vals = np.amin(voxel_coords[:3, :], axis=1)
    shift = np.array(_min_absolute_value(max_vals, min_vals))
    transform_direction = np.array([[1., 0, 0, -shift[0]], [0, 1., 0, -shift[1]], [0, 0, 1., -shift[2]], [0, 0, 0, 1.]])
    inv_transform_direction = np.array([[1., 0, 0, shift[0] - 1], [0, 1., 0, shift[1] - 1],
                                        [0, 0, 1., shift[2] - 1], [0, 0, 0, 1.]])
    voxel_coords = transform_direction @ voxel_coords

    voxel_volume_size = np.around(max_vals - min_vals).astype(int)
    voxels = np.zeros((voxel_volume_size + np.ones_like(voxel_volume_size)))

    voxel_coords = np.around(voxel_coords[:3, :].T).astype(int)
    for x, y, z in voxel_coords:
        voxels[x, y, z] = 1
    voxels = np.pad(voxels, 1, constant_values=0)
    t1 = datetime.datetime.now()
    boundary_voxels = find_boundary_by_force(voxels)
    t2 = datetime.datetime.now()

    print("Execution time for boundary searching is: " + str(t2 - t1))

    boundary_voxel_coords = np.vstack(np.where(boundary_voxels == 1))
    boundary_voxel_coords = np.pad(boundary_voxel_coords, ((0, 1), (0, 0)), constant_values=1)

    boundary_coords = ((transform_scale @ (inv_transform_direction @ boundary_voxel_coords))[:3, :]).T

    # verts, faces, normals, values = measure.marching_cubes(voxels, step_size=1)
    # verts_4d = np.hstack((verts, np.ones((verts.shape[0], 1))))
    # verts_coords = (transform_scale @ inv_transform_direction @ verts_4d.T)

    # best_approximations = find_best_approximations(voxel_coords, verts, normals)
    # approx_4d = np.hstack((best_approximations, np.ones((best_approximations.shape[0], 1))))
    # approx_coords = (transform_scale @ inv_transform_direction @ approx_4d.T)

    """
    result = None
    for vertx in verts_coords.T:
        arr_bc = np.broadcast_to(vertx.T, np.shape(coords_4d))
        indx = np.argmin(np.linalg.norm((arr_bc[:, :3] - coords_4d[:, :3]), axis=1), axis=0)
        
        if result is None:
            result = np.array([coords[indx]])
        else:
            result = np.concatenate((result, np.array([coords[indx]])), axis=0)

        coords_4d[indx] = [np.inf, np.inf, np.inf, np.inf]
    """

    # result = [coords[np.argmin(np.abs(arr - coords))] for arr in verts_coords]
    # result = np.vsplit((verts_coords.T)[:, :3], np.shape(verts_coords)[1])

    if plot:
        xcoords_bound = [arr[0] for arr in boundary_coords]
        ycoords_bound = [arr[1] for arr in boundary_coords]
        zcoords_bound = [arr[2] for arr in boundary_coords]

        xcoords = [arr[0] for arr in coords]
        ycoords = [arr[1] for arr in coords]
        zcoords = [arr[2] for arr in coords]

        fig = plt.figure(1)
        ax = plt.axes(projection='3d')
        ax.scatter(xcoords, ycoords, zcoords)
        ax.scatter(xcoords_bound, ycoords_bound, zcoords_bound)

        # ig2 = plt.figure(2, figsize=(10, 10))
        # ax2 = fig2.add_subplot(111, projection='3d')
        # ax2.set_xlim(0, 24)
        # ax2.set_ylim(0, 20)
        # ax2.set_zlim(0, 32)

        # mesh = Poly3DCollection(verts[faces])
        # mesh.set_edgecolor('k')
        # ax2.add_collection3d(mesh)

        # plt.tight_layout()
        plt.show()

    check_bc_coord_equality(boundary_coords, coords)


def _min_absolute_value(a1, a2):
    stacked = np.vstack((a1, a2))
    indices = np.argmin(np.absolute(stacked), axis=0)
    return [int(stacked[indices[0], 0]), int(stacked[indices[1], 1]), int(stacked[indices[2], 2])]


# def _point_plane_dist(point, normal, vertex):
#   d = -normal[0] * vertex[0] - normal[1] * vertex[1] - normal[2] * vertex[2]
#  dist = (normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2] + d) / np.linalg.norm(normal)
# return dist


# @jit
def find_best_approximations(coords, verts, normals):
    best_approximations = np.zeros_like(normals)

    for j, (vertex, normal) in enumerate(zip(verts, normals)):
        distances = np.zeros(np.shape(coords)[0])

        for ii, point in enumerate(coords):
            d = -normal[0] * vertex[0] - normal[1] * vertex[1] - normal[2] * vertex[2]
            distances[ii] = (normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2] + d) / np.linalg.norm(
                normal)

        # if np.any(distances) > 0:  # The first-hand choice should be points that are outside the boundary defined by the
        # planes
        # minind = np.argmin(np.where(distances > 0, distances, np.inf))
        # best_approximations[j] = coords[minind]
        # coords = np.delete
        # else:
        minind = np.argmin(np.absolute(distances))
        best_approximations[j] = coords[minind]
        coords[minind] = [np.inf, np.inf, np.inf]
    return best_approximations


def find_boundary_by_erosion(voxels):
    voxel_copy_z = np.copy(voxels)
    for z_layer in range(np.shape(voxels)[2]):
        # voxel_copy_z[:, :, z_layer] = scipy.ndimage.binary_erosion(voxel_copy_z[:, :, z_layer])
        diff_x = np.diff(voxel_copy_z[:, :, z_layer], axis=1, n=1)
        diff_x = np.pad(diff_x, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        diff_y = np.diff(voxel_copy_z[:, :, z_layer], axis=0, n=1)
        diff_y = np.pad(diff_y, ((0, 1), (0, 0)), mode='constant', constant_values=0)

        tmp = np.ma.masked_values(np.abs(diff_x), 1).mask + np.ma.masked_values(np.abs(diff_y), 1).mask
        voxel_copy_z[:, :, z_layer] += diff_y
        voxel_copy_z[:, :, z_layer] += diff_x

        # diff = diff_x + diff_y.T
    # borders_z = voxels - voxel_copy_z


def find_boundary_by_force(voxels):
    dim = np.shape(voxels)
    boundary = np.zeros_like(voxels)
    for i in range(0, dim[0]):
        for j in range(0, dim[1]):
            for k in range(0, dim[2]):
                if voxels[i, j, k] == 1:
                    if np.any(voxels[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2] == 0.):
                        boundary[i, j, k] = 1
                else:
                    continue

    return boundary


@jit
def check_bc_coord_equality(boundary, coords):
    for bc in boundary:
        compare_bc_to_coords = np.array([(np.linalg.norm(coord - bc) < 1e-10) for coord in coords])
        if not np.any(compare_bc_to_coords):
            print(bc)
            raise RuntimeWarning("Warning: One or several points of the boundary cannot be found in the original list "
                                 "of points defining the body!" )
    print("Passed boundary equality check!")


vals, coords = vtk_to_numpy("/home/philip/Desktop/grain_stress_5.vtu")
alphashape(coords, plot=True)
