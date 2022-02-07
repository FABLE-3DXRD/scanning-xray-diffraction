import datetime
import math
from pyevtk.hl import pointsToVTK
import numpy as np
import vtkmodules.vtkIOXML as vtk_xml
import vtkmodules.util.numpy_support as vtk_np
from numpy import ndarray
import matplotlib.pyplot as plt
from numba import jit
from scipy import ndimage


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

    return np.array(values), coords


def alphashape(coords, values, nlayers=1, plot=False):
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
        plt.show()

    _check_bc_coord_equality(boundary_coords, coords)
    keys = [tuple(np.round(c, 5)) for c in coords]
    data_dict = dict(zip(keys, values.T))
    components = ["XX", "YY", "ZZ", "YZ", "XZ", "XY"]
    bounday_data = np.vstack([data_dict[tuple(np.round(bc, 5))] for bc in boundary_coords])
    pointsToVTK("/home/philip/Desktop/grain_boundary", boundary_coords[:, 0], boundary_coords[:, 1],
                boundary_coords[:, 2], dict(zip(components, np.ascontiguousarray(bounday_data.T))))
    return boundary_coords


def _min_absolute_value(a1, a2):
    stacked = np.vstack((a1, a2))
    indices = np.argmin(np.absolute(stacked), axis=0)
    return [int(stacked[indices[0], 0]), int(stacked[indices[1], 1]), int(stacked[indices[2], 2])]


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
def _check_bc_coord_equality(boundary, coords):
    for bc in boundary:
        compare_bc_to_coords = np.array([(np.linalg.norm(coord - bc) < 1e-10) for coord in coords])
        if not np.any(compare_bc_to_coords):
            print(bc)
            raise RuntimeWarning("Warning: One or several points of the boundary cannot be found in the original list "
                                 "of points defining the body!")
    print("Passed boundary equality check!")


def project_to_plane(boundary_coordinates, coordinates, values, projection_function, plot=False):
    cms = (np.sum(boundary_coordinates, axis=0) / np.shape(boundary_coordinates)[0])
    coordinates = np.pad(coordinates, ((0, 0), (0, 1)), constant_values=1)
    boundary_coordinates = np.pad(boundary_coordinates, ((0, 0), (0, 1)), constant_values=1)

    move_grain_to_cms = np.array([[1, 0, 0, -cms[0]], [0, 1, 0, -cms[1]], [0, 0, 1, -cms[2]], [0, 0, 0, 1]])
    coordinates = (move_grain_to_cms @ coordinates.T).T
    boundary_coordinates = (move_grain_to_cms @ boundary_coordinates.T).T[:, :3]
    # Note: from here on the grain is centered at the origin, but the stresses/strains have not changed direction!
    sphere_coords = [_carthesian_to_spherical(bc, np.array([0, 0, 0])) for bc in boundary_coordinates]
    projected_coords = projection_function(sphere_coords)

    if plot:
        xcoords = [arr[0] for arr in coordinates]
        ycoords = [arr[1] for arr in coordinates]
        zcoords = [arr[2] for arr in coordinates]

        xcoords_projected = [arr[0] for arr in projected_coords]
        ycoords_projected = [arr[1] for arr in projected_coords]

        fig = plt.figure(1)
        ax = plt.axes(projection='3d')
        ax.scatter(xcoords, ycoords, zcoords)

        fig = plt.figure(2)
        ax = plt.axes()
        ax.scatter(xcoords_projected, ycoords_projected)
        # ax.set_ylim([0, 3000])
        plt.show()

    components = ["XX", "YY", "ZZ", "YZ", "XZ", "XY"]
    keys = [tuple(np.round(c, 5)) for c in coordinates[:, :3]]
    data_dict = dict(zip(keys, values.T))

    bounday_data = np.vstack([data_dict[tuple(np.round(bc, 5))] for bc in boundary_coordinates])
    pointsToVTK("/home/philip/Desktop/grain_flat_wink", projected_coords[:, 0], projected_coords[:, 1],
                np.zeros(np.shape(projected_coords)[0]), dict(zip(components, np.ascontiguousarray(bounday_data.T))))


def _find_by_vector_norm(assortment, target):
    for indx in enumerate(assortment):
        if np.linalg.norm(assortment[indx] - target) < 1e-10:
            return indx


def _carthesian_to_spherical(coord, cms):
    location = coord - cms
    r = np.linalg.norm(location)
    phi = math.asin(location[2] / r)
    lamda = math.atan2(location[1], location[0])
    return (r, phi, lamda)


def _mercator(spherical_coordinates):
    x = np.array([sphere_coord[0] * sphere_coord[2] for sphere_coord in spherical_coordinates])
    y = np.array([sphere_coord[0] * np.log(np.tan(np.pi / 4 + sphere_coord[1] / 2))
                  for sphere_coord in spherical_coordinates])
    return np.vstack((x, y)).T


def _winkel_III(spherical_coordinates):
    delta = [np.arccos(np.cos(sphere_coord[1]) * np.cos(sphere_coord[2]/2)) for sphere_coord in spherical_coordinates]
    lamda = [np.arccos(np.sin(sphere_coord[1])/np.sin(delta[i])) for i, sphere_coord in enumerate(spherical_coordinates)]

    x = [0.5 * sphere_coord[0] * (2 * np.sign(sphere_coord[2]) * delta[i] * np.sin(lamda[i]) + sphere_coord[2] * np.cos(np.deg2rad(40)))
         for i, sphere_coord in enumerate(spherical_coordinates)]
    y = [0.5 * sphere_coord[0] * (delta[i] * np.cos(lamda[i]) + sphere_coord[1]) for i, sphere_coord in enumerate(spherical_coordinates)]
    """
    phi_1 = np.arccos(2 / np.pi)
    alpha = [np.arccos(np.cos(sphere_coord[1]) * np.cos(sphere_coord[2] / 2)) for sphere_coord in spherical_coordinates]
    x = [0.5 * (sphere_coord[2] * np.cos(phi_1) + (2 * np.cos(sphere_coord[1]) * np.sin(sphere_coord[2] / 2)) /
                np.sinc(alpha[i])) for i, sphere_coord in enumerate(spherical_coordinates)]
    y = [0.5 * (sphere_coord[1] + (np.sin(sphere_coord[1]) / np.sinc(alpha[i])))
         for i, sphere_coord in enumerate(spherical_coordinates)]
    """
    return np.vstack((x, y)).T


vals, coords = vtk_to_numpy("/home/philip/Desktop/grain_stress_5.vtu")
boundary = alphashape(coords, vals, plot=False)
project_to_plane(boundary, coords, vals, _winkel_III)
