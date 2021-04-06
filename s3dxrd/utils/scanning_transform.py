
from __future__ import print_function

""" Modified from ImageD11 transform.py by Axel Henningsson 2021
    This is only good for the scanning_3DXRD scenario when the t_y==0 and 
    t_z==0 for all reflections, the idea is to allow for per measurement
    grain cms positions to be passed via t_x,t_y,t_z params."""

import numpy as np
from ImageD11 import gv_general
from numpy import radians, degrees

def cross_product_2x2(a, b):
    """ returns axb for two len(3) vectors a,b"""
    assert len(a) == len(b) == 3
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])

def detector_rotation_matrix(tilt_x, tilt_y, tilt_z):
    """
    Return the tilt matrix to apply to peaks
    tilts in radians
    typically applied to peaks rotating around beam center
    """
    r1 = np.array([[np.cos(tilt_z), -np.sin(tilt_z), 0],  # note this is r.h.
                   [np.sin(tilt_z), np.cos(tilt_z), 0],
                   [0,    0, 1]], np.float)
    r2 = np.array([[np.cos(tilt_y), 0, np.sin(tilt_y)],
                   [0, 1,   0],
                   [-np.sin(tilt_y), 0, np.cos(tilt_y)]], np.float)
    r3 = np.array([[1,          0,       0],
                   [0,  np.cos(tilt_x), -np.sin(tilt_x)],
                   [0,  np.sin(tilt_x), np.cos(tilt_x)]], np.float)
    r2r1 = np.dot(np.dot(r3, r2), r1)
    return r2r1

def compute_xyz_lab(peaks,
                    y_center=0., y_size=0., tilt_y=0.,
                    z_center=0., z_size=0., tilt_z=0.,
                    tilt_x=0.,
                    distance=0.,
                    # detector_orientation=((1,0),(0,1)),
                    o11=1.0, o12=0.0, o21=0.0, o22=-1.0,
                    **kwds):
    """
    Peaks is a 2 d array of x,y
    yc is the centre in y
    ys is the y pixel size
    ty is the tilt around y
    zc is the centre in z
    zs is the z pixel size
    tz is the tilt around z
    dist is the sample - detector distance
    detector_orientation is a matrix to apply to peaks arg to get
    ImageD11 convention
         (( 0, 1),( 1, 0)) for ( y, x)
         ((-1, 0),( 0, 1)) for (-x, y)
         (( 0,-1),(-1, 0)) for (-y,-x)
      etc...
    """
    assert len(peaks) == 2, "peaks must be a 2D array"
    # Matrix for the tilt rotations
    r2r1 = detector_rotation_matrix(tilt_x, tilt_y, tilt_z)
    # Peak positions in 3D space
    #  - apply detector orientation
    peaks_on_detector = np.array(peaks)
    peaks_on_detector[0, :] = (peaks_on_detector[0, :] - z_center) * z_size
    peaks_on_detector[1, :] = (peaks_on_detector[1, :] - y_center) * y_size
    #
    detector_orientation = [[o11, o12], [o21, o22]]
    # logging.debug("detector_orientation = "+str(detector_orientation))
    flipped = np.dot(np.array(detector_orientation, np.float),
                     peaks_on_detector)
    #
    vec = np.array([np.zeros(flipped.shape[1]),  # place detector at zero,
                    # sample at -dist
                    flipped[1, :],             # x in search, frelon +z
                    flipped[0, :]], np.float)     # y in search, frelon -y
    # Position of diffraction spots in 3d space after detector tilts about
    # the beam centre on the detector
    rotvec = np.dot(r2r1, vec)
    # Now add the distance (along x)
    rotvec[0, :] = rotvec[0, :] + distance
    return rotvec


def compute_tth_eta(peaks,
                    y_center=0., y_size=0., tilt_y=0.,
                    z_center=0., z_size=0., tilt_z=0.,
                    tilt_x=0.,
                    distance=0.,
                    # detector_orientation=((1,0),(0,1)),
                    o11=1.0, o12=0.0, o21=0.0, o22=-1.0,
                    t_x=0.0, t_y=0.0, t_z=0.0,
                    omega=None,  # == phi at chi=90
                    wedge=0.0,  # Wedge == theta on 4circ
                    chi=0.0,  # == chi - 90
                    **kwds):  # spare args are ignored
    """
    0/10 for style
    """

    peaks_xyz = compute_xyz_lab(
        peaks,
        y_center=y_center, y_size=y_size, tilt_y=tilt_y,
        z_center=z_center, z_size=z_size, tilt_z=tilt_z,
        tilt_x=tilt_x,
        distance=distance,
        # detector_orientation=((1,0),(0,1)),
        o11=o11, o12=o12, o21=o21, o22=o22)

    tth, eta = compute_tth_eta_from_xyz(
        peaks_xyz,
        t_x=t_x, t_y=t_y, t_z=t_z,
        omega=omega,
        wedge=wedge,
        chi=chi)

    return tth, eta


def compute_tth_eta_from_xyz(peaks_xyz, omega,
                             t_x=0.0, t_y=0.0, t_z=0.0,
                             #       == phi at chi=90
                             wedge=0.0,  # Wedge == theta on 4circ
                             chi=0.0,  # == chi - 90
                             **kwds):  # last line is for laziness -
    """
    Peaks is a 3 d array of x,y,z peak co-ordinates
    crystal_translation is the position of the grain giving rise
    to a diffraction spot
                 in x,y,z ImageD11 co-ordinates
                 x,y with respect to axis of rotation and or beam centre ??
                 z with respect to beam height, z centre
    omega data needed if crystal translations used
    """
    assert len(peaks_xyz) == 3
    # Scattering vectors
    if omega is None:
        s1 = peaks_xyz
    else:
        # scattering_vectors
        if len(omega) != len(peaks_xyz[0]):
            raise Exception(
                "omega and peaks arrays must have same number of peaks")
        s1 = peaks_xyz - compute_grain_origins(omega, wedge, chi,
                                               t_x, t_y, t_z)
    # CHANGED to HFP convention 4-9-2007
    eta = np.degrees(np.arctan2(-s1[1, :], s1[2, :]))
    s1_perp_x = np.sqrt(s1[1, :] * s1[1, :] + s1[2, :] * s1[2, :])
    tth = np.degrees(np.arctan2(s1_perp_x, s1[0, :]))
    return tth, eta


def compute_xyz_from_tth_eta(tth, eta, omega,
                             t_x=0.0, t_y=0.0, t_z=0.0,
                             #       == phi at chi=90
                             wedge=0.0,  # Wedge == theta on 4circ
                             chi=0.0,  # == chi - 90
                             **kwds):  # last line is for laziness -
    """
    Given the tth, eta and omega, compute the xyz on the detector

    crystal_translation is the position of the grain giving rise
    to a diffraction spot
                 in x,y,z ImageD11 co-ordinates
                 x,y with respect to axis of rotation and or beam centre ??
                 z with respect to beam height, z centre

    omega data needed if crystal translations used
    """
    # xyz = unit vectors along the scattered vectors
    xyz = np.zeros((3, tth.shape[0]), np.float)
    rtth = np.radians(tth)
    reta = np.radians(eta)
    xyz[0, :] =  np.cos(rtth)
    #  eta = np.degrees(np.arctan2(-s1[1, :], s1[2, :]))
    xyz[1, :] = -np.sin(rtth) * np.sin(reta)
    xyz[2, :] =  np.sin(rtth) * np.cos(reta)

    # Find vectors in the fast, slow directions in the detector plane
    pks = np.array([(1, 0),
                    (0, 1),
                    (0, 0) ], np.float).T
    dxyzl = compute_xyz_lab(pks, **kwds)
    # == [xpos, ypos, zpos] shape (3,n)
    #
    # This was based on the recipe from Thomas in Acta Cryst ...
    #  ... Modern Equations of ...

    ds = dxyzl[:,0] - dxyzl[:,2]  # 1,0 in plane is (1,0)-(0,0)
    df = dxyzl[:,1] - dxyzl[:,2]  # 0,1 in plane
    dO = dxyzl[:,2]               # origin pixel

    # Cross products to get the detector normal
    # Thomas uses an inverse matrix, but then divides out the determinant anyway
    det_norm = np.cross( ds, df )

    # Scattered rays on detector normal
    norm = np.dot( det_norm, xyz )
    # Check for divide by zero
    msk = (norm == 0)
    needmask = False
    if msk.sum()>0:
        norm += msk
        needmask = True

    # Intersect ray on detector plane
    sc = np.dot( np.cross( df, dO ), xyz ) / norm
    fc = np.dot( np.cross( dO, ds ), xyz ) / norm

    if (t_x != 0) or (t_y != 0) or (t_z != 0):
        go = compute_grain_origins(omega,
                                   wedge=wedge, chi=chi,
                                   t_x=t_x, t_y=t_y, t_z=t_z)
        # project these onto the detector face to give shifts
        sct = (  xyz * np.cross( df, go.T ).T ).sum(axis=0) / norm
        fct = (  xyz * np.cross( go.T, ds ).T ).sum(axis=0) / norm
        sc -= sct
        fc -= fct

    if needmask:
        fc = np.where( msk, 0, fc )
        sc = np.where( msk, 0, sc )

    return fc, sc


def compute_grain_origins(omega, wedge=0.0, chi=0.0,
                          t_x=0.0, t_y=0.0, t_z=0.0):
    """
    # print "Using translations t_x %f t_y %f t_z %f"%(t_x,t_y,t_z)
    # Compute positions of grains
    # expecting tx, ty, tz for each diffraction spot
    #
    # g =  R . W . k
    #  g - is g-vector w.r.t crystal
    #  k is scattering vector in lab
    #  so we want displacement in lab from displacement in sample
    #  shift =  W-1  R-1 crystal_translation
    #
    # R = ( cos(omega) , sin(omega), 0 )
    #     (-sin(omega) , cos(omega), 0 )
    #     (         0  ,         0 , 1 )
    #
    # W = ( cos(wedge) ,  0  ,  sin(wedge) )
    #     (         0  ,  1  ,          0  )
    #     (-sin(wedge) ,  0  ,  cos(wedge) )
    #
    # C = (         1  ,          0  ,       0     ) ??? Use eta0 instead
    #     (         0  ,   cos(chi)  ,  sin(chi)   )  ??? Use eta0 instead
    #     (         0  ,  -sin(chi)  ,  cos(chi)   )  ??? Use eta0 instead
    """
    w = np.radians(wedge)
    WI = np.array([[np.cos(w),         0, -np.sin(w)],
                   [0,           1,         0],
                   [np.sin(w),         0,  np.cos(w)]], np.float)
    c = np.radians(chi)
    CI = np.array([[1,            0,         0],
                   [0,     np.cos(c), -np.sin(c)],
                   [0,     np.sin(c),  np.cos(c)]], np.float)
    t = np.zeros((3, omega.shape[0]), np.float)  # crystal translations
    # Rotations in reverse order compared to making g-vector
    # also reverse directions. this is trans at all zero to
    # current setting. gv is scattering vector to all zero
    om_r = np.radians(omega)
    # This is the real rotation (right handed, g back to k)

    # Modified for translating sample, input t_x,t_y,t_z are
    # now precomputed scattering positions.
    #---------------------------------------------------
    t[0, :] = t_x
    t[1, :] = t_y
    t[2, :] = t_z
    
    # import matplotlib.pyplot as plt
    # plt.plot(t_x, t_y, "bo")
    # plt.plot(np.cos(om_r) * t_x - np.sin(om_r) * t_y, np.sin(om_r) * t_x + np.cos(om_r) * t_y, "ro")
    # plt.xlim([-600,600])
    # plt.ylim([-600,600])
    # plt.show()

    #old code:
    #t[0, :] = np.cos(om_r) * t_x - np.sin(om_r) * t_y
    #t[1, :] = np.sin(om_r) * t_x + np.cos(om_r) * t_y
    #t[2, :] = t_z
    #---------------------------------------------------


    if chi != 0.0:
        c = np.cos(np.radians(chi))
        s = np.sin(np.radians(chi))
        u = np.zeros(t.shape, np.float)
        u[0, :] = t[0, :]
        u[1, :] = c * t[1, :] + -s * t[2, :]
        u[2, :] = s * t[1, :] + c * t[2, :]
        t = u
    if wedge != 0.0:
        c = np.cos(np.radians(wedge))
        s = np.sin(np.radians(wedge))
        u = np.zeros(t.shape, np.float)
        u[0, :] = c * t[0, :] + -s * t[2, :]
        u[1, :] = t[1, :]
        u[2, :] = s * t[0, :] + c * t[2, :]
        t = u
    return t

def compute_k_vectors(tth, eta, wvln):
    """
    generate k vectors - scattering vectors in laboratory frame

    NOTE: These are the G_l vectors. not k vectors -... - Axel Henningsson 2021
          These do not have length 2pi/wvlv but some other scaling.
    """
    tth = np.radians(tth)
    eta = np.radians(eta)
    c = np.cos(tth / 2)  # cos theta
    s = np.sin(tth / 2)  # sin theta
    ds = 2 * s / wvln
    k = np.zeros((3, tth.shape[0]), np.float)
    # x - along incident beam
    k[0, :] = -ds * s  # this is negative x
    # y - towards door
    k[1, :] = -ds * c * np.sin(eta)  # CHANGED eta to HFP convention 4-9-2007
    # z - towards roof
    k[2, :] = ds * c * np.cos(eta)

    return k


def compute_g_vectors(tth,
                      eta,
                      omega,
                      wvln,
                      wedge=0.0,
                      chi=0.0):
    """
    Generates spot positions in reciprocal space from
      twotheta, wavelength, omega and eta
    Assumes single axis vertical
    ... unless a wedge angle is specified
    """
    k = compute_k_vectors(tth, eta, wvln)
#    print k[:,0]
    return compute_g_from_k(k, omega, wedge, chi)


def compute_g_from_k(k, omega, wedge=0, chi=0):
    """
    Compute g-vectors with cached k-vectors
    """
    om = np.radians(omega)
    # G-vectors - rotate k onto the crystal axes
    g = np.zeros((3, k.shape[1]), np.float)
    t = np.zeros((3, k.shape[1]), np.float)
    #
    # g =  R . W . k where:
    # R = ( cos(omega) , sin(omega), 0 )
    #     (-sin(omega) , cos(omega), 0 )
    #     (         0  ,         0 , 1 )
    #
    # W = ( cos(wedge) ,  0  ,  sin(wedge) )
    #     (         0  ,  1  ,          0  )
    #     (-sin(wedge) ,  0  ,  cos(wedge) )
    #
    # C = (         1  ,         0  ,      0     )
    #     (         0  ,  cos(chi)  , sin(chi)   )
    #     (         0  , -sin(chi)  , cos(chi)   )
    #
    if wedge != 0.0:
        c = np.cos(np.radians(wedge))
        s = np.sin(np.radians(wedge))
        t[0, :] = c * k[0, :] + s * k[2, :]
        t[1, :] = k[1, :]
        t[2, :] = -s * k[0, :] + c * k[2, :]
        k = t.copy()
    if chi != 0.0:
        c = np.cos(np.radians(chi))
        s = np.sin(np.radians(chi))
        t[0, :] = k[0, :]
        t[1, :] = c * k[1, :] + s * k[2, :]
        t[2, :] = -s * k[1, :] + c * k[2, :]
        k = t.copy()
    # This is the reverse rotation (left handed, k back to g)
    g[0, :] = np.cos(om) * k[0, :] + np.sin(om) * k[1, :]
    g[1, :] = -np.sin(om) * k[0, :] + np.cos(om) * k[1, :]
    g[2, :] = k[2, :]
    return g


def uncompute_g_vectors(g, wavelength, wedge=0.0, chi=0.0):
    """
    Given g-vectors compute tth,eta,omega
    assert uncompute_g_vectors(compute_g_vector(tth,eta,omega))==tth,eta,omega
    """
    if wedge == chi == 0:
        post = None
    else:
        post = gv_general.wedgechi( wedge=wedge, chi=chi )
    omega1, omega2, valid = gv_general.g_to_k(
        g, wavelength,axis=[0,0,-1], pre=None, post=post )
    # we know g, omega. Compute k as ... ?
    if post is None:
        pre = None
    else:
        pre = gv_general.chiwedge( wedge=wedge, chi=chi ).T
    k_one = gv_general.k_to_g( g, omega1, axis=[0,0,1],
                               pre = pre, post=None)
    k_two = gv_general.k_to_g( g, omega2, axis=[0,0,1],
                               pre = pre, post=None)
    #
    # k[1,:] = -ds*c*sin(eta)
    # ------    -------------   .... tan(eta) = -k1/k2
    # k[2,:] =  ds*c*cos(eta)
    #
    eta_one = np.arctan2(-k_one[1, :], k_one[2, :])
    eta_two = np.arctan2(-k_two[1, :], k_two[2, :])
    #
    #
    ds = np.sqrt(np.sum(g * g, 0))
    s = ds * wavelength / 2.0  # sin theta
    tth = np.degrees(np.arcsin(s) * 2.) * valid
    eta1 = np.degrees(eta_one) * valid
    eta2 = np.degrees(eta_two) * valid
    omega1 = omega1 * valid
    omega2 = omega2 * valid
    return tth, [eta1, eta2], [omega1, omega2]