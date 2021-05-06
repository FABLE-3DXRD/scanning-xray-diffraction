import numpy as np
import matplotlib.pyplot as plt
from ImageD11 import parameters, grain, transform
import sys
from . import grain_fitter
from . import scanning_transform
from . import reconstruct_grainshapes


def map_peaks( flt, grains, params, omegastep, hkltol, nmedian, ymin, ystep, number_y_scans ):
    omslop = omegastep/2.
    
    tth, eta, gve = initiate_cols( flt, params, omslop )

    for i,gr in enumerate(grains):
        assign_peaks_to_grain( gr, gve, flt, params, nmedian,  hkltol )
    discard_overlaping_spots( grains, gve, flt, params, nmedian, hkltol )

    init_assigned_peaks=0
    for i,gr in enumerate(grains):
        init_assigned_peaks += np.sum(gr.mask)

    x = np.zeros((len(grains,)))        # current grain centroid x xoordinates
    y = np.zeros((len(grains,)))        # current grain centroid y coordinates
    assigned_peaks = 0
    prev_assigned_peaks = 0

    
    for j in range(2):

        # Compute current grain centroids and assign peaks
        for i,gr in enumerate(grains):

            # Grain centroids can be computed based on the sinogram.
            _, gr.sino, gr.recon = reconstruct_grainshapes.FBP_grain( gr, flt, ymin, ystep, omegastep, number_y_scans )

            tth, eta, gve = grain_fitter.get_peak_quantities( flt, params, gr )
            assign_peaks_to_grain( gr, gve, flt, params, nmedian,  hkltol )
        discard_overlaping_spots( grains, gve, flt, params, nmedian,  hkltol )

        # Update grain ubi based on the newly assigned peak sets
        for i,gr in enumerate(grains):
            grain_fitter.fit_one_grain( gr, flt, params )

    tth, eta, gve = update_cols_per_grain( flt, params, omslop, grains )

def initiate_cols( flt, pars, OMSLOP, weights=True  ):
    tth, eta, gve = grain_fitter.get_peak_quantities( flt, pars )
    flt.addcolumn( tth  , "tth" )
    flt.addcolumn( eta , "eta" )
    flt.addcolumn( gve[0], "gx" )
    flt.addcolumn( gve[1], "gy" )
    flt.addcolumn( gve[2], "gz" )

    if weights:
        wtth, weta, womega = grain_fitter.estimate_weights( pars, flt, OMSLOP )
        flt.addcolumn( wtth, "wtth" )
        flt.addcolumn( weta, "weta" )
        flt.addcolumn( womega, "womega" )

    return tth, eta, gve

def assign_peaks( grains, gve, flt, pars, nmedian, hkltol ):
    """
    Assign peaks to grains for fitting
    - each grain chooses the spots it likes
    - overlapping spots (chosen by more than 1 grain) are removed
    - fit outliers are removed abs(median err) > nmedian
    Fills in grain.mask for each grain
    """
    for i, g in enumerate(grains):
        hkl = np.dot( g.ubi, gve )
        hkli = np.round( hkl )
        drlv = hkli - hkl
        drlv2 = (drlv*drlv).sum(axis=0)
        g.mask = drlv2 < hkltol*hkltol
    discard_overlaping_spots( grains, gve, flt, pars, nmedian,  hkltol )

def update_mask( mygr, flt, pars, nmedian ):
    """
    Remove nmedian*median_error outliers from grains assigned peaks
    This routine fills in mygr.mask and mygr.hkl
    """
    # obs data for this grain
    tthobs = flt.tth[ mygr.mask ]
    etaobs = flt.eta[ mygr.mask ]
    omegaobs = flt.omega[ mygr.mask ]
    gobs = np.array( (flt.gx[mygr.mask], flt.gy[mygr.mask], flt.gz[mygr.mask]) )
    # hkls for these peaks
    hklr = np.dot( mygr.ubi, gobs )
    hkl  = np.round( hklr )
    # Now get the computed tth, eta, omega
    etasigns = np.sign( etaobs )
    mygr.hkl = hkl.astype(int)
    mygr.etasigns = etasigns
    ub = np.linalg.inv(mygr.ubi)
    tthcalc, etacalc, omegacalc = grain_fitter.calc_tth_eta_omega( ub, hkl, pars, etasigns )
    # update mask on outliers
    dtth = (tthcalc - tthobs)
    deta = (etacalc - etaobs)
    domega = (omegacalc - omegaobs)
    msk  = abs( dtth ) > np.median( abs( dtth   ) ) * nmedian
    msk |= abs( deta ) > np.median( abs( deta   ) ) * nmedian
    msk |= abs( domega)> np.median( abs( domega ) ) * nmedian
    allinds = np.arange( flt.nrows )
    mygr.mask[ allinds[mygr.mask][msk] ] = False
    return msk.astype(int).sum()

def assign_peaks_to_grain( gr, gve, flt, pars, nmedian, hkltol):
    """
    Assign peaks to grains for fitting
    - each grain chooses the spots it likes
    Fills in grain.mask for each grain
    """

    # For each grain we compute the hkl integer labels
    hkl = np.dot( gr.ubi, gve )
    hkli = np.round( hkl )
    # Error on these:
    drlv = hkli - hkl
    drlv2 = (drlv*drlv).sum(axis=0)
    # Tolerance to assign to a grain is rather poor

    gr.mask = drlv2 < hkltol*hkltol # g.mask is a boolean declaration of all peaks that can belong to grain g

def discard_overlaping_spots( grains, gve, flt, pars, nmedian, hkltol ):
    """
    Iterate over all grains and discard any spots choosen by more than one grain
    """

    overlapping = np.zeros( flt.nrows, dtype=bool )
    for i in range(len(grains)):
        for j in range(i+1,len(grains)):
            overlapping |= grains[i].mask & grains[j].mask

    for i, g in enumerate(grains):
        g.mask &= ~overlapping

        while 1:
            ret = update_mask( g, flt, pars, nmedian )
            if ret == 0:
                break

def update_cols_per_grain( flt, pars, OMSLOP, grains):
    """
    update the twotheta, eta, g-vector columns fill in weighting estimates for fitting ubi matrices.
    """

    # Make sure no grains are trying to use the same peaks.
    for j,gr1  in enumerate(grains):
        for i,gr2 in enumerate(grains):
            if i!=j: assert np.sum(gr1.mask*gr2.mask)==0

    # print( "Peaks indexed: ", np.sum([gr.mask for gr in grains])  )
    # print( "Total nbr peaks: ", flt.nrows)

    # Fill flt columns by iterating over the grains
    for i,gr in enumerate(grains):
        peak_pos = [flt.sc[gr.mask], flt.fc[gr.mask]]
        omega = flt.omega[ gr.mask ]
        dty = flt.dty[gr.mask]

        pars.parameters['t_x'] = grain_fitter.get_cms_along_beam( gr.recon, 
                                                                  omega, 
                                                                  dty, 
                                                                  pars.parameters['ymin'], 
                                                                  pars.parameters['ystep'] )
        pars.parameters['t_y'] = np.zeros((len(dty),))
        pars.parameters['t_z'] = np.zeros((len(dty),))

        tth, eta = scanning_transform.compute_tth_eta( peak_pos, omega=omega, **pars.parameters )
        gve = scanning_transform.compute_g_vectors( tth, eta, omega,
                                            pars.get('wavelength'),
                                            wedge=pars.get('wedge'),
                                            chi=pars.get('chi') )
        pars.parameters['t_x'] = None
        pars.parameters['t_y'] = None
        pars.parameters['t_z'] = None

        flt.tth[ gr.mask ] = tth
        flt.eta[ gr.mask ] = eta
        if flt.wtth.any(): 
            # Compute the relative tth, eta, omega errors ...
            wtth, weta, womega = grain_fitter.estimate_weights( pars, flt, OMSLOP, gr )
            flt.wtth[ gr.mask ] = wtth
            flt.weta[ gr.mask ] = weta
            flt.womega[ gr.mask ] = womega
        flt.gx[ gr.mask ] = gve[0]
        flt.gy[ gr.mask ] = gve[1]
        flt.gz[ gr.mask ] = gve[2]

    return flt.tth, flt.eta, np.array([flt.gx, flt.gy, flt.gz])