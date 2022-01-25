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
    C = np.reshape(C, (6, 6, 1))
    C_rot = np.zeros(C.shape)
    for i in range(C.shape[2]):
        C_rot[:, :, i] = (M[:, :, i] @ C[:, :, i] @ np.transpose(M[:, :, i]))
    return C_rot


def _get_rotation_matrix(U):
    """Return the rotation matrix, R, that corresponds to rotating the stiffness tensor by the rotation defined by U

    R*eps_bar contains the same values in vector format where eps_bar is 6x1 stack of eps uniqie values as

    Args:
        U (:obj:`numpy array`): Orientation/rotation tensors, ```shape=(Nx3x3)```

    Returns:
        (:obj:`numpy array`) of ```shape=(Nx6x6)``` rotation matrix.

    """
    M = np.array([[U[:, 0, 0] ** 2, U[:, 0, 1] ** 2, U[:, 0, 2] ** 2,       2 * U[:, 0, 1] * U[:, 0, 2], 2 * U[:, 0, 2] * U[:, 0, 0], 2 * U[:, 0, 0] * U[:, 0, 1]],
                  [U[:, 1, 0] ** 2, U[:, 1, 1] ** 2, U[:, 1, 2] ** 2,       2 * U[:, 1, 1] * U[:, 1, 2], 2 * U[:, 1, 2] * U[:, 1, 0], 2 * U[:, 1, 0] * U[:, 1, 1]],
                  [U[:, 2, 0] ** 2, U[:, 2, 1] ** 2, U[:, 2, 2] ** 2,       2 * U[:, 2, 1] * U[:, 2, 2], 2 * U[:, 2, 2] * U[:, 2, 0], 2 * U[:, 2, 1] * U[:, 2, 1]],
                  [U[:, 1, 0] * U[:, 2, 0], U[:, 1, 1] * U[:, 2, 1], U[:, 1, 2] * U[:, 2, 2],       U[:, 1, 1] * U[:, 2, 2] + U[:, 1, 2] * U[:, 2, 1],  U[:, 1, 0] * U[:, 2, 2] + U[:, 1, 2] * U[:, 2, 0],  U[:, 1, 1] * U[:, 2, 0] + U[:, 1, 0] * U[:, 2, 1]],
                  [U[:, 2, 0] * U[:, 0, 0], U[:, 2, 1] * U[:, 0, 1], U[:, 2, 2] * U[:, 0, 2],       U[:, 0, 1] * U[:, 2, 2] + U[:, 0, 2] * U[:, 2, 1],  U[:, 0, 2] * U[:, 2, 0] + U[:, 0, 0] * U[:, 2, 2],  U[:, 0, 0] * U[:, 2, 1] + U[:, 0, 1] * U[:, 2, 0]],
                  [U[:, 0, 0] * U[:, 1, 0], U[:, 0, 1] * U[:, 1, 1], U[:, 0, 2] * U[:, 1, 2],       U[:, 0, 1] * U[:, 1, 2] + U[:, 0, 2] * U[:, 1, 1],  U[:, 0, 2] * U[:, 1, 0] + U[:, 0, 0] * U[:, 1, 2],  U[:, 0, 0] * U[:, 1, 1] + U[:, 0, 1] * U[:, 1, 0]]])
    return M


def calculate_stress(strains, grains):
    pass