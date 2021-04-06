import numpy as np
from scipy.optimize import leastsq
from ImageD11 import parameters, grain, transform
import copy
from . import scanning_transform
from skimage.transform import warp
import matplotlib.pyplot as plt

def fit_one_grain( gr, flt, pars):
    """
    Uses scipy.optimize to fit a single grain
    """
    args = flt, pars, gr
    ub = np.linalg.inv(gr.ubi)
    x0 = ub.ravel().copy()
    xf, cov_v, info, mesg, ier = leastsq(
        calc_teo_fit, x0, args, full_output=True)
    ub = xf.copy()
    ub.shape = 3, 3
    ubi = np.linalg.inv(ub)

    gr.set_ubi(ubi)

def calc_teo_fit( ub, flt, pars, gr, return_concatenated=True):
    """
    Function for refining ub using tth, eta, omega data
    ub is the parameter array to fit
    flt is all the data
    pars in the diffractometer geometry to get tthcalc, etacalc, omegacalc
    gr is the grain holding the peak assignments
    flt.wtth, weta, wometa = weighting functions for tth vs eta vs omega errors
    """
    UB = np.array(ub)
    UB.shape = 3, 3

    tthcalc, etacalc, omegacalc = calc_tth_eta_omega(
        UB, gr.hkl, pars, gr.etasigns)

    dtth = (flt.tth[gr.mask] - tthcalc) * flt.wtth[gr.mask]
    deta = (flt.eta[gr.mask] - etacalc) * flt.weta[gr.mask]
    domega = (flt.omega[gr.mask] - omegacalc) * flt.womega[gr.mask]
    if return_concatenated:
        return np.concatenate((dtth, deta, domega))
    else:
        return dtth, deta, domega

def calc_tth_eta_omega( ub, hkls, pars, etasigns):
    """
    Predict the tth, eta, omega for each grain
    ub = ub matrix (inverse ubi)
    hkls = peaks to predict
    pars = diffractometer info (wavelength, rotation axis)
    etasigns = which solution for omega/eta to choose (+y or -y)
    """
    g = np.dot(ub, hkls)

    tthcalc, eta2, omega2 = transform.uncompute_g_vectors(g,  pars.get('wavelength'),
                                                            wedge=pars.get('wedge'),
                                                            chi=pars.get('chi'))
    # choose which solution (eta+ or eta-)
    e0 = np.sign(eta2[0]) == etasigns
    etacalc = np.where(e0, eta2[0], eta2[1])
    omegacalc = np.where(e0, omega2[0], omega2[1])
    return tthcalc, etacalc, omegacalc

def get_peak_quantities( flt, pars, grain=None ):
    """
    Compute twotheta, eta, g-vector for given grain. Note that the entire peak
    set will be placed at the grain cms, no respect is taken to the grian peak mask.
    if grain is None, it is assumed that all scattering occurs from the lab origin.
    """
    if grain is None:
        pars.parameters['t_x'] = np.zeros(flt.nrows)
        pars.parameters['t_y'] = np.zeros(flt.nrows)
        pars.parameters['t_z'] = np.zeros(flt.nrows)
    else:
        pars.parameters['t_x'] = get_cms_along_beam( grain.recon, 
                                                    flt.omega, 
                                                    flt.dty, 
                                                    pars.parameters['ymin'], 
                                                    pars.parameters['ystep'] )       
        pars.parameters['t_y'] = np.zeros(flt.nrows)
        pars.parameters['t_z'] = np.zeros(flt.nrows)

    tth, eta = scanning_transform.compute_tth_eta( [flt.sc, flt.fc], omega=flt.omega, **pars.parameters)

    gve = scanning_transform.compute_g_vectors(tth, eta, flt.omega,
                                        pars.get('wavelength'),
                                        wedge=pars.get('wedge'),
                                        chi=pars.get('chi'))

    pars.parameters['t_x'] = None
    pars.parameters['t_y'] = None
    pars.parameters['t_z'] = None

    return tth, eta, gve

def estimate_weights( pars, flt, OMSLOP, g=None):
    distance = pars.get('distance')
    pixelsize = (pars.get('y_size') + pars.get('z_size')) / 2.0
    # 1 pixel - high energy far detector approximation
    if g:
        wtth = np.ones(np.sum(g.mask)) / np.degrees(pixelsize / distance)
        weta = wtth * np.tan(np.radians(flt.tth[g.mask]))
        womega = np.ones(np.sum(g.mask))/OMSLOP
    else:
        wtth = np.ones(flt.nrows) / np.degrees(pixelsize / distance)
        weta = wtth * np.tan(np.radians(flt.tth))
        womega = np.ones(flt.nrows)/OMSLOP

    return wtth, weta, womega


def get_cms_along_beam( recon, omega, dty, ymin, ystep, stepsize=0.5 ):
    """Compute grain centroid component along lab - x axis (beam direction) based on a 2d pixel reconstruction.

    This function rotates the reconstructed grain shapes by provided omega angles and computes the centroid 
    along the beam. To speed up computation, only every stepsize degrees is computed for, and the input angles
    are mapped to closest angle.

    Args:
        recon (:obj:`numpy array`): Pixelated reconstruction of grain shape, ```shape=(m,m)```.
        omega (:obj:`numpy array`): Projection angles in units of degrees, ```shape=(n,)```.
        ystep (:obj:`float`): Pixel size of reconstruction
        stepsize (:obj:`float`): degree stepsize resolution only these rotations are used (for speed).

    Returns:
        ```cms_along_beam``` 1d ```numpy array``` with centroid x coordinates, ```shape=(n,)```.

    """

    assert recon.shape[0]==recon.shape[1], "Only square recons will work"
    
    cms_weights = np.linspace(0,recon.shape[0],recon.shape[0])
    angles = np.arange(np.min(omega), np.max(omega) + stepsize, stepsize)
    cos_a, sin_a = np.cos(np.radians(angles)), np.sin(np.radians(angles))
    center = recon.shape[0] // 2
    image = np.abs(recon)
    rotated = []

    # Rotate reconstruction in steps of stepsize dgrs.
    for i, (c,s) in enumerate(zip(cos_a, sin_a)):
        R = np.array([[c, s, -center * (c + s - 1)],
                        [-s, c, -center * (c - s - 1)],
                        [0, 0, 1]])
        rotated.append( warp(image, R, clip=False) )

    # Map to closest rotated reconstruction.
    cmsx = np.zeros((len(omega),))
    iy = np.round( (dty - ymin) / ystep ).astype(int)
    for i,(om,yindx) in enumerate(zip(omega,iy)):
        indx  = np.argmin( np.abs( angles-om ) )
        cmsx[i] = ( (np.sum( rotated[indx][:,yindx]*cms_weights  ) / np.sum( rotated[indx][:,yindx] )) - center ) * ystep

    return cmsx