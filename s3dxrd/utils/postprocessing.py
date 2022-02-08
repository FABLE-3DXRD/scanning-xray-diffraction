import datetime
import math
import numpy
from pyevtk.hl import pointsToVTK
import numpy as np
import vtkmodules.vtkIOXML as vtk_xml
import vtkmodules.util.numpy_support as vtk_np
from numpy import ndarray
import matplotlib.pyplot as plt
from numba import jit
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from s3dxrd.utils.stiffness import _vec_to_tens


def vtk_to_numpy(vtkfile, plot=False):
    """
    Import point cloud data stored in a VTK file to a list of Numpy arrays.

    :param plot: Plot the reconstructed data as a point cloud.
    :type plot: bool
    :param vtkfile: The path of the file containing the original data as an unstructured grid.
    :type vtkfile: str
    :return: Numpy array containing the data provided in the input file.
    :rtype: tuple[ndarray, ndarray]

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


def boundary(coords, values, file, nlayers=1, normal_values=None, plot=False):
    # TODO: Implement multi-layer alpha shape calculation.
    """
    Calculate the alpha shape (the concave hull) for a point cloud consisting of a given set of
    three-dimensional coordinates. The code presumes that the coordinates are given in microns and that the
    measurements are taken 25 microns apart.

    :param file: The name of the .vtu file where the boundary will be output.
    :type file: str
    :param coords: List of coordinates in x, y and z for the different points of the point cloud.
    :type coords:  ndarray
    :param nlayers: Number of layers in the alpha shape, Defaults to one (the outermost layer).
    :type nlayers: int
    :param plot: Toggle plotting of the alpha shape as a point cloud and as a tessellated mesh body. Defaults to False.
    :type plot: bool
    :param values: Values of the different normal and shear stresses or strains to be assigned to the boundary points.
    :type values: ndarray
    :param normal_values: Parameter used for testing. If the boundary points are already known,
        the values for a quantity in the direction of the surface normal can be supplied here
        for plotting at the boundary points.
    :return: Tuple containing ndarrays of the coordinates of the points in the point cloud corresponding to the boundary
        of the input point cloud, the data at the boundary points, the matrix used for transforming the voxel
        representation of the grain into the sample coordinate system, the matrix used to move the grain from the centre
        of the coordinate system to the original position of the grain and the voxelated grain representation.
    :rtype: tuple[ndarray, ndarray, ndarray, ndarray, ndarray]
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
    boundary_voxels = find_boundary(voxels)
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
    boundary_data = np.vstack([data_dict[tuple(np.round(bc, 5))] for bc in boundary_coords])

    # For testing purposes only, requires previous calculation of the boundary points.
    if normal_values is not None:
        components.append("Normal")
        boundary_data = np.hstack((boundary_data, numpy.reshape(normal_values, (-1, 1))))

    pointsToVTK(file, boundary_coords[:, 0], boundary_coords[:, 1],
                boundary_coords[:, 2], dict(zip(components, np.ascontiguousarray(boundary_data.T))))

    return boundary_coords, boundary_data, transform_scale, inv_transform_direction, voxels


def find_boundary(voxels):
    """
    Finds the boundary voxels using a linear search method.

    :param voxels: A binarised voxel representation of the grains where 1 corresponds to the voxel being part of the
    grains, and 0 means that the voxel is not part of the grain.
    :type voxels: ndarray
    :return: Voxelated representation of the grain boundary.
    :rtype: ndarray
    """
    dim = np.shape(voxels)
    boundary = np.zeros_like(voxels)
    for i in range(1, dim[0] - 1):
        for j in range(1, dim[1] - 1):
            for k in range(1, dim[2] - 1):
                if voxels[i, j, k] == 1:
                    if np.any(voxels[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2] == 0.):
                        boundary[i, j, k] = 1
                else:
                    continue

    return boundary


@jit
def _check_bc_coord_equality(boundary, coords):
    """
    Private method for checking that the calculated boundary points are a subset of the entire pointcloud.

    :param boundary: The boundary points
    :type boundary: ndarray
    :param coords: The coordinates defining the pointcloud.
    :type coords: ndarray
    :raise: RuntimeWarning
    :return: None
    """
    for bc in boundary:
        compare_bc_to_coords = np.array([(np.linalg.norm(coord - bc) < 1e-10) for coord in coords])
        if not np.any(compare_bc_to_coords):
            print(bc)
            raise RuntimeWarning("Warning: One or several points of the boundary cannot be found in the original list "
                                 "of points defining the body!")
    print("Passed boundary equality check!")


def project_to_plane(boundary_coordinates, coordinates, values, projection_function, file, normal_values=None,
                     plot=False):
    """
    Project tensorial quantities calculated for the boundary of a 3D spheriod onto a 2D surface.

    :param boundary_coordinates: The coordinates for the boundary of the 3D surface.
    :type boundary_coordinates: ndarray
    :param coordinates: The coordinates of the pointcloud defining the 3D volume where the boundary originates.
    :type coordinates: ndarray
    :param values: The tensorial quantities that will be projected at the 2D surface.
    :type values: ndarray
    :param projection_function: A function for converting coordinates defined using radius, latitude and longitude into
        coordinates in the XY-plane.
    :type projection_function: any
    :param file: The filename where the resulting .vtu file will be output.
    :type file: str
    :param normal_values: Values in the direction of the surface normal to the body, calculated at the boundary points.
    :type normal_values: ndarray
    :param plot: Toggle plotting.
    :type plot: bool
    :return: None
    """
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
    boundary_data = np.vstack([data_dict[tuple(np.round(bc, 5))] for bc in boundary_coordinates])

    if normal_values is not None:
        components.append("Normal")
        boundary_data = np.hstack((boundary_data, numpy.reshape(normal_values, (-1, 1))))

    pointsToVTK(file, projected_coords[:, 0], projected_coords[:, 1],
                np.zeros(np.shape(projected_coords)[0]), dict(zip(components, np.ascontiguousarray(boundary_data.T))))


def _carthesian_to_spherical(coord, cms):
    """
    Private method for converting Carthesian coordinates into coordinates based on radius, latitude and longitude.

    :param coord: A vector containing the Carthesian coordinates of a point.
    :type coord: ndarray
    :param cms: The centroid of the pointcloud in Carthesian coordinates.
    :type cms: ndarray
    :return: Tuple contaning the radial coordinate, the latitude and the longitude.
    :rtype: tuple
    """
    location = coord - cms
    r = np.linalg.norm(location)
    phi = math.asin(location[2] / r)
    lamda = math.atan2(location[1], location[0])
    return r, phi, lamda


def _mercator(lat_long):
    """
    Calculate the 2D X and Y coordinates from a set of coordinates based on radius, latitude and longitude using the
    Mercator projection.
    :param lat_long: The coordinates of the points to be projected expressed as radius, latitude and longitude.
    :type lat_long: list[tuple]
    :return: The projected coordinates in the XY-plane.
    :rtype: ndarray
    """
    x = np.array([coord[0] * coord[2] for coord in lat_long])
    y = np.array([coord[0] * np.log(np.tan(np.pi / 4 + coord[1] / 2))
                  for coord in lat_long])
    return np.vstack((x, y)).T


def _winkel_III(lat_long):
    """
    Calculate the 2D X and Y coordinates from a set of coordinates based on radius, latitude and longitude using the
    Winkel III projection.
    :param lat_long: The coordinates of the points to be projected expressed as radius, latitude and longitude.
    :type lat_long: list[tuple]
    :return: The projected coordinates in the XY-plane.
    :rtype: ndarray
    """
    delta = [np.arccos(np.cos(coord[1]) * np.cos(coord[2] / 2)) for coord in lat_long]
    lamda = [np.arccos(np.sin(coord[1]) / np.sin(delta[i]))
             for i, coord in enumerate(lat_long)]

    x = [0.5 * coord[0] * (2 * np.sign(coord[2]) * delta[i] * np.sin(lamda[i]) + coord[2] *
                                  np.cos(np.deg2rad(40))) for i, coord in enumerate(lat_long)]
    y = [0.5 * coord[0] * (delta[i] * np.cos(lamda[i]) + coord[1])
         for i, coord in enumerate(lat_long)]
    return np.vstack((x, y)).T


def find_normals_mc(voxels, boundary_points, boundary_data, scale_mat, inv_dir_mat, plot=False):
    """
    Find the normals of the surface created from a set of boundary points, using the Marching Cubes algorithm. The
    normals are calculated as an average of the normals at the surface vertices that are closest to the boundary points.

    :param voxels: Three-dimensional array providing a voxelated representation of the grain.
    :type voxels: ndarray
    :param boundary_points: The Carthesian coordinates of the boundary points.
    :type boundary_points: ndarray
    :param boundary_data: The stress/strain vectors at the boundary points.
    :type boundary_data: ndarray
    :param scale_mat: The matrix used to scale the voxels into the original coordinates of the point cloud.
    :type scale_mat: ndarray
    :param inv_dir_mat: The matrix which moves the grain centroid from the origin of the coordinate system to its
        position in the sample coordinate system.
    :type inv_dir_mat: ndarray
    :param plot: Toggle plotting.
    :type plot: bool
    :return: Tuple containing the average normals, the values of boundary_data projected on the normals at the
        boundary points and the scalar product of the normals calculated using the Marching Cubes algorithm with the
        corresponding vectors directed from the grain centroid to the boundary points, i.e. an alternative
        approximation of the normal vectors.
    :rtype: tuple[ndarray, ndarray, ndarray]
    """
    verts, faces, normals, values = measure.marching_cubes(voxels, step_size=1)
    verts_4d = np.hstack((verts, np.ones((verts.shape[0], 1))))
    normals_4d = np.hstack((normals, np.ones((normals.shape[0], 1))))

    verts_coords = (scale_mat @ (inv_dir_mat @ verts_4d.T)).T[:, :3]
    normal_coords = ((np.linalg.inv(scale_mat @ inv_dir_mat)).T @ normals_4d.T).T[:, :3]
    for k in range(np.shape(normal_coords)[0]):
        normal_coords[k] = np.divide(normal_coords[k], np.linalg.norm(normal_coords[k]))

    @jit
    def _search_for_normals(boundary_points):
        avg_normals = np.zeros_like(boundary_points)
        for i, bp in enumerate(boundary_points):
            # Note that these dimensions fo not match since verts_coords is nverts x 3 and bp is 1 x 3. Numpy
            # broadcasting should take care of this automatically.
            diffs = np.zeros(np.shape(verts_coords)[0])
            for j in range(np.shape(verts_coords)[0]):
                diffs[j] = np.linalg.norm(verts_coords[j, :] - bp)

            mindiff = np.inf
            diffinds = np.empty(1, dtype=numpy.int64)
            for j, diff in enumerate(diffs):
                if diff < mindiff:
                    diffinds = np.array([j])
                    mindiff = diff
                elif np.abs(diff - mindiff) < 1e-8:
                    diffinds = np.append(diffinds, j)
            avg_normals[i] = np.sum(normal_coords[diffinds], axis=0) / np.shape(diffinds)[0]
            avg_normals[i] = avg_normals[i] / np.linalg.norm(avg_normals[i])

        return avg_normals

    avg_normals = _search_for_normals(boundary_points)
    normal_values = np.array([avg_normals[row] @ _vec_to_tens(boundary_data[row]) @ avg_normals[row].T
                              for row in range(np.shape(boundary_data)[0])])

    cms = (np.sum(boundary_points, axis=0) / np.shape(boundary_points)[0])
    norm_from_cms = (boundary_points - cms)
    for l, t in enumerate(norm_from_cms):
        norm_from_cms[l] = t / np.linalg.norm(t)

    deviation = np.array([np.dot(norm_from_cms[i], avg_normals[i]) for i in range(np.shape(avg_normals)[0])])
    for t in norm_from_cms:
        print(np.linalg.norm(t))

    if plot:
        xverts = [arr[0] for arr in verts_coords]
        yverts = [arr[1] for arr in verts_coords]
        zverts = [arr[2] for arr in verts_coords]

        xbound = [arr[0] for arr in boundary_points]
        ybound = [arr[1] for arr in boundary_points]
        zbound = [arr[2] for arr in boundary_points]

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # ax.scatter(xverts, yverts, zverts, c='g', s=1)

        mesh = Poly3DCollection(verts_coords[faces])
        mesh.set_edgecolor('k')
        mesh.set_alpha(0.5)
        ax.add_collection3d(mesh)
        # ax.scatter(xbound, ybound, zbound, c='r')

        ax.quiver(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
                  avg_normals[:, 0], avg_normals[:, 1], avg_normals[:, 2], length=30, color='y')

        ax.quiver(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
                  norm_from_cms[:, 0], norm_from_cms[:, 1], norm_from_cms[:, 2], length=300, color='g')
        plt.show()
    return avg_normals, normal_values, deviation


vals, coords = vtk_to_numpy("/home/philip/Desktop/grain_stress_5.vtu")
boundary_coords, boundary_data, transform_scale, inv_transform_direction, voxels = boundary(coords, vals,
                                                                                            "/home/philip/Desktop/grain_boundary",
                                                                                            plot=False)
avg_normals, normal_stresses, deviation = find_normals_mc(voxels, boundary_coords, boundary_data, transform_scale,
                                                          inv_transform_direction)
boundary(coords, vals, normal_values=normal_stresses, plot=False)
project_to_plane(boundary_coords, coords, vals, _winkel_III, normal_values=normal_stresses)
