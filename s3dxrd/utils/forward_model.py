import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from . import analytical_strain_fields as asf

def project_strain( strain, Xgrid, Ygrid, Zgrid, entry, exit, nhat , kappa, L, nsegs, nfirst ):
    '''Forward model average strain along ray paths given an analytical
    strain tensor field and some ray geometries.

    Input:
        strain : 
        entry  : is a 3 by N array of the coordinates where the ray enters the sample (each col is x, y, z)
        exit   : is a 3 by N array of the coordinates where the ray leaves the sample (each col is x, y,z)
        nhat   : is a 3 by N array of directions of each ray (each col is a unit vector)
        kappa  : is a 3 by N array that defines the direction of strain measured (each col is a unit vector)
        L      : is a 1 by N array of the total irradiated lengths of each ray (ie distance between entry and exit)
        nsegs  : is a 1 by N array of number of segments for each ray
    Returns:
        Y      : is N by 1 array of forward modeled measurement values
    '''
    Y = np.zeros( (L.shape[1],) )

    for i in range( nfirst ):

        y = 0
        indx = np.argmin( np.abs( Zgrid.flatten()-entry[2, i] ) )
        mask = ( Zgrid.flatten()==Zgrid.flatten()[indx] )
        xvec = Xgrid.flatten()[mask]
        yvec = Ygrid.flatten()[mask]

        eps = [ s.flatten()[mask] for s in strain]
        strain_tens_func = asf.discrete_to_analytic( eps, xvec, yvec )

        for j in range( int(nsegs[0,i]) ):

            p1 = entry[3*j:3*j + 3,i] 
            p2 = exit[3*j:3*j + 3,i]
            l = np.linalg.norm(p2-p1)

            def func(s):
                epsilon = strain_tens_func(p1 + s*nhat[:,i])
                return kappa[:,i].T.dot( epsilon ).dot( kappa[:,i] )

            y += quad(func, 0, l)[0]

        Y[i] = y/L[0,i]
    
    return Y

        