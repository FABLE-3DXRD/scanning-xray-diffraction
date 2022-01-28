import numpy as np


def stiff_mat():
    """
    Returns the stiffness matrix for α-silicon dioxide, as defined in Voigt notation.
    :returns: The 6x6 stiffness matrix for α-silicon dioxide
    """
    C11, C33, C44, C66, C12, C13, C14 = 86.99, 106.39, 58.12, 40.12, 6.75, 12.17, 17.99

    C = np.zeros((6, 6))
    C[0, :4] = [C11, C12, C13, C14]
    C[1, 1:4] = [C11, C13, -C14]
    C[2, 2] = C33
    C[3, 3] = C44
    C[4, 4:6] = [C44, C14]
    C[5, 5] = C66

    C = C + np.tril(C.T, -1)
    C = C*(10**9) # Convert to GPa
    return C

def rotate_stiffness(U, C):
    """Rotate a series of 6x6 "tensor matrices" according to a series of 3x3 rotation matrices.

    Args:
        U (:obj:`numpy array`): Orientation/rotation tensors, ```shape=(Nx3x3)```
        C (:obj:`numpy array`): Stiffness matrices, ```shape=(Nx6x6)```

    Returns:
        (:obj:`numpy array`) of ```shape=(Nx6x6)``` rotated compliance matrices.

    """
    M = _get_rotation_matrix(U)
    C_rot = (M @ C @ np.transpose(M))

    return C_rot


def _get_rotation_matrix(U):
    """Return the rotation matrix, R, that corresponds to rotating the stiffness tensor by the rotation defined by U

    R*eps_bar contains the same values in vector format where eps_bar is 6x1 stack of eps uniqie values as

    Args:
        U (:obj:`numpy array`): Orientation/rotation tensors, ```shape=(Nx3x3)```

    Returns:
        (:obj:`numpy array`) of ```shape=(Nx6x6)``` rotation matrix.

    """
    M = np.array([[U[0, 0] ** 2, U[0, 1] ** 2, U[0, 2] ** 2,       2 * U[0, 1] * U[0, 2], 2 * U[0, 2] * U[0, 0], 2 * U[0, 0] * U[0, 1]],
                  [U[1, 0] ** 2, U[1, 1] ** 2, U[1, 2] ** 2,       2 * U[1, 1] * U[1, 2], 2 * U[1, 2] * U[1, 0], 2 * U[1, 0] * U[1, 1]],
                  [U[2, 0] ** 2, U[2, 1] ** 2, U[2, 2] ** 2,       2 * U[2, 1] * U[2, 2], 2 * U[2, 2] * U[2, 0], 2 * U[2, 0] * U[2, 1]],

                  [U[1, 0] * U[2, 0], U[1, 1] * U[2, 1], U[1, 2] * U[2, 2],       U[1, 1] * U[2, 2] + U[1, 2] * U[2, 1],  U[1, 0] * U[2, 2] + U[1, 2] * U[2, 0],  U[1, 1] * U[2, 0] + U[1, 0] * U[2, 1]],
                  [U[2, 0] * U[0, 0], U[2, 1] * U[0, 1], U[2, 2] * U[0, 2],       U[0, 1] * U[2, 2] + U[0, 2] * U[2, 1],  U[0, 2] * U[2, 0] + U[0, 0] * U[2, 2],  U[0, 0] * U[2, 1] + U[0, 1] * U[2, 0]],
                  [U[0, 0] * U[1, 0], U[0, 1] * U[1, 1], U[0, 2] * U[1, 2],       U[0, 1] * U[1, 2] + U[0, 2] * U[1, 1],  U[0, 2] * U[1, 0] + U[0, 0] * U[1, 2],  U[0, 0] * U[1, 1] + U[0, 1] * U[1, 0]]])
    return M


def calculate_stress_by_matrix_rotation(wlsq_strain, U):
    # Get the stiffness matrix as measured in the grain coordinate system
    C = stiff_mat()
    # Rotate the stiffness matrix by the grain orientation matrix
    C = rotate_stiffness(U, C)
    # Stack the strain vectors into a matrix, where each row contains the strain components for a certain element in
    # the mesh which the stress will be plotted on. Make an empty matrix for the stress vectors.
    strain_mat = np.column_stack(wlsq_strain)
    stress_mat = np.zeros_like(strain_mat)

    # Exract a row from the strain matrix, multiply the shear strain components by 2 to obtain the engineeering shear
    # strain which is compatible with the Voigt notation.
    for i in range(np.size(strain_mat, 0)):
        strain_vector = strain_mat[i, :]
        strain_vector[3:6] *= 2
        # Apply the stiffness matrix to get the stress vectors and stack the stress vectors in a matrix.
        stress_mat[i, :] = C @ strain_vector
    # Split the stress matrix to give it the same format as wlsq_strains.
    wlsq_stress = np.hsplit(stress_mat, 6)

    for i, arr in enumerate(wlsq_stress):
        wlsq_stress[i] = arr.reshape((-1))
        print("Processing array " + str(i))
    return wlsq_stress

def calculate_stress_by_vector_rotation(wlsq_strain, U):
    # Here the stresses will be calculated using a different method where the strain vectors are rotated into the grain
    # coordinate system where the stiffness matrix is applied and then the corresponding strain vector is rotated back
    # by solving a system of equations.

    # Get the stiffness matrix as measured in the grain coordinate system
    C = stiff_mat()

    # Stack the strain vectors into a matrix, where each row contains the strain components for a certain element in
    # the mesh which the stress will be plotted on. Make an empty matrix for the stress vectors.
    strain_mat = np.column_stack(wlsq_strain)
    stress_mat = np.zeros_like(strain_mat)

    for i in range(np.size(strain_mat, 0)):
        strain_vector = strain_mat[i, :]
        # Transform the strain_vector to the grain coordinate system.
        strain_tensor = _vec_to_tens(strain_vector)
        grain_strain_tensor = U.T @ strain_tensor @ U

        # Convert the grain strain vector to Voigt notation.
        grain_strain_vector = _tens_to_vec(grain_strain_tensor)
        grain_strain_vector[3:6] *= 2

        # Calculate the stress in the grain coordinate system and apply the U matrix to transform the strain back to
        # the sample coordinate system.
        grain_stress_vector = C @ grain_strain_vector

        # Convert the stress vector to the sample coordinate system.
        grain_stess_tensor = _vec_to_tens(grain_stress_vector)
        sample_stress_tensor = U @ grain_stess_tensor @ U.T
        sample_stress_vector = _tens_to_vec(sample_stress_tensor)

        stress_mat[i, :] = sample_stress_vector

    # Split the stress matrix to give it the same format as wlsq_strains.
    wlsq_stress = np.hsplit(stress_mat, 6)
    for i, arr in enumerate(wlsq_stress):
        wlsq_stress[i] = arr.reshape((-1))
        print("Processing array " + str(i))
    return wlsq_stress

def _vec_to_tens(vec):
    tens = np.row_stack(np.array([[vec[0], vec[5], vec[4]], [vec[5], vec[1], vec[3]], [vec[4], vec[3], vec[2]]]))
    return tens

def _tens_to_vec(tens):
    vec = np.array([tens[0, 0], tens[1, 1], tens[2, 2], tens[1, 2], tens[0, 2], tens[0, 1]])
    return vec