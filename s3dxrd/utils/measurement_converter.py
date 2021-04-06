import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import iradon, radon
from xfab import tools
import torch

def vectors_to_torch(vectors):
    """Convert numpy arrays in a vectors dictionary to torch double format.

    Args:
        vectors (:obj:`dict`): vectors dictionary as specified by s3dxrd.measurements.Id11.peaks_to_vectors()
    
    Returns:
        (:obj:`dict`): dictionary with same fields as ```vectors``` but with torch tensors replacing
            numby arrays.
    """
    vectors["Y"]      =  torch.from_numpy(vectors["Y"]).double()
    vectors["entry"]  =  torch.from_numpy(vectors["entry"]).double()
    vectors["exit"]   =  torch.from_numpy(vectors["exit"]).double()
    vectors["nhat"]   =  torch.from_numpy(vectors["nhat"]).double()
    vectors["kappa"]  =  torch.from_numpy(vectors["kappa"]).double()
    vectors["L"]      =  torch.from_numpy(vectors["L"]).double()
    vectors["nsegs"]  =  torch.from_numpy(vectors["nsegs"]).double()
    vectors["sig_m"]  =  torch.from_numpy(vectors["sig_m"]).double()
    return vectors

def sind( a ): return np.sin( np.radians( a ) )

def cosd( a ): return np.cos( np.radians( a ) )

def omega_matrix( om ):
    return np.array([[cosd(om),-sind(om),0],[sind(om),cosd(om),0],[0,0,1]])

def get_k(tth, eta, wavelength):
    return (2*np.pi/wavelength)*np.array([1, 0, 0])

def get_k_prime(tth, eta, wavelength):
    k1 = cosd(tth)
    k2 = -sind(tth)*sind(eta)
    k3 = sind(tth)*cosd(eta)
    return (2*np.pi/wavelength)*np.array([k1,k2,k3])

def get_Q(tth, eta, wavelength):
    k = get_k(tth, eta, wavelength)
    kp = get_k_prime(tth, eta, wavelength)
    return k - kp

def normal_omega(tth, eta, wavelength, omega):
    '''
    Return strained planes normal in omega coordinate system
    '''
    Qlab = get_Q(tth, eta, wavelength)
    Qomega = np.dot( np.linalg.inv( omega_matrix( omega ) ), Qlab )
    return -Qomega/np.linalg.norm(Qomega)

def taylor_strain(G_w, G_0):
    '''Strain in direction nhat by using Taylor expansion
    of the rewritten Laue equations.
    '''
    s = 1 - ( G_w.dot(G_0) /  G_0.dot(G_0) )
    return s

def strain(B, h, k, l, tth, wavelength):
    #G_original = np.sqrt((h*a_rec)**2 + (k*b_rec)**2 + (l*c_rec)**2)

    G_original = np.dot(B,np.array([h,k,l]))

    # Elm. of Modern X-ray P. pp. 157
    d_original = 2*np.pi / np.linalg.norm(G_original)

    # Bragg's law
    m = np.round(2*d_original*sind(tth/2.)/wavelength)
    d_measured = (m*wavelength)/(2*sind(tth/2.))

    return m, d_measured, d_original, (d_measured-d_original)/d_original

def weight(d_measured, d_original, strain, tth, wavelength, bragg_order, distance, pixelsize):
    '''
    Calculate weights based on resolution concerns, to be used in As=m taking
    into account that the rows of A are differently accurate measurements.

    UNITS:
          wavelength  -  arbitrary unit A
          d_original  -  arbitrary unit A
          d_measured  -  arbitrary unit A
          distance    -  arbitrary unit B
          pixelsize   -  arbitrary unit B
    '''

    tth_rad = np.radians( tth )
    r = np.tan( tth_rad )*distance # peak location radius
    dtth_rad = np.arctan( (r + pixelsize )/distance ) - tth_rad # angular width of a pixel
    d_rdr = ( (bragg_order*wavelength)/(2*np.sin( (tth_rad+dtth_rad)/2. ) ) )
    eps = (d_rdr-d_original)/d_original # strain at radius r + dr
    w = abs(1 / (strain - eps) )
    assert strain>eps
    
    return w

def uniq( vals ):
    d = {}
    newvals = []
    for v in vals:
        if v not in d:
            d[v]=0
            newvals.append(v)
    return newvals

def convert_measurements( params, grain, flt, ymin, ystep, omegastep ):

    distance = params.get('distance')
    pixelsize = ( params.get('y_size') + params.get('z_size') ) / 2.0
    wavelength = params.get('wavelength')
    cell = [params.get('cell__a'),params.get('cell__b'),params.get('cell__c'),params.get('cell_alpha'),params.get('cell_beta'),params.get('cell_gamma')]

    B_0 = tools.form_b_mat(cell)

    keys = [ (hkl[0], hkl[1], hkl[2], int(s))
             for hkl, s in zip(grain.hkl.T , grain.etasigns)]

    uni = uniq(keys)
    akeys = np.array( keys )
    strains = []
    directions = []
    all_omegas = []
    dtys = []
    all_tths = []
    all_etas = []
    weights = []
    all_intensity = []
    all_sc = []
    all_Gws = []
    all_hkl = []

    for refi,u in enumerate(uni):

        # h==h, k==k, l==l, sign==sign
        mask = (akeys == u).astype(int).sum(axis=1) == 4
        tths = flt.tth[grain.mask][mask]
        etas = flt.eta[grain.mask][mask]
        omegas = flt.omega[grain.mask][mask]
        scs = flt.sc[grain.mask][mask]
        detector_y_pos = flt.dty[ grain.mask ][mask]
        intensity = flt.sum_intensity[grain.mask][mask]
        scs = flt.sc[grain.mask][mask]
        G_ws = np.array( (flt.gx[grain.mask][mask], flt.gy[grain.mask][mask], flt.gz[grain.mask][mask]) ).T
        h = u[0]
        k = u[1]
        l = u[2]
        
        for sc, dty, tth, eta, om, I, sc, G_w in zip( scs, detector_y_pos, tths, etas, omegas, intensity, scs, G_ws):

            all_hkl.append( [h,k,l] )
            bragg_order, d_measured, d_original, eps = strain(B_0, h, k, l, tth, wavelength)


            G_0 = grain.u.dot( B_0.dot( np.array([h,k,l]) ) )
            Qlab = get_Q(tth, eta, wavelength)
            Qomega = -np.dot( np.linalg.inv( omega_matrix( om ) ), Qlab )
            eps_taylor = taylor_strain(Qomega, G_0) #<= to use Henningssons Taylor
            
            strains.append( eps_taylor )

            #print(eps, eps_taylor)
            #strains.append( eps ) #<= to use Poulsens
            
            #directions.append( G_0/np.linalg.norm(G_0) ) #<= this approximation is probably equally good
            directions.append( normal_omega(tth, eta, wavelength, om) )
            all_omegas.append( om )
            all_tths.append( tth )
            all_etas.append( eta )
            dtys.append( dty )
            all_intensity.append( I )
            all_sc.append( sc )
            all_Gws.append(G_w)
            #plt.scatter(dty, eps)
            #plt.title(r'$\omega$ = '+str(om))
            
            weights.append( weight( d_measured, d_original, eps, tth, wavelength, bragg_order, distance, pixelsize ) )
        #plt.show()
    measurements = {}
    measurements['strain']    = np.array(strains)
    measurements['kappa']     = np.array(directions)
    measurements['omega']     = np.array(all_omegas)
    measurements['dty']       = np.array(dtys)
    measurements['sig_m']     = 1./np.array(weights)
    measurements['tth']       = np.array(all_tths)
    measurements['eta']       = np.array(all_etas)
    measurements['intensity'] = np.array(all_intensity)
    measurements['sc']        = np.array(all_sc)
    measurements['Gomega']    = np.asarray(all_Gws)
    measurements['hkl']       = np.asarray(all_hkl)

    if 0:
        sort_index = np.argsort(measurements['tth'])
        plt.plot(measurements['tth'][sort_index]/2., measurements['sig_m'][sort_index])
        plt.xlabel(r'$\theta$ [dgr]')
        plt.ylabel(r'$\sigma_m$')
        plt.grid(True)
        plt.show()
    return measurements

