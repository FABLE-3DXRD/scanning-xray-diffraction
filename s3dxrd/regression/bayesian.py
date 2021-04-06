import torch
import numpy as np

def defineFourierBasis(m,lx,ly,lz, sig_f, nhat=None, subset=None):
    """Defines a set of Fourier basis functions used in the bayesian regression.

    Args:
        m (:obj:`torch.tensor`): number of basis functions along each axis
        lx (:obj:`torch.tensor`): lengthscale along x axis
        ly (:obj:`torch.tensor`): lengthscale along y axis
        lz (:obj:`torch.tensor`): lengthscale along z axis
        sig_f (:obj:`torch.tensor`): prior signal uncertainty
        nhat (:obj:`torch.tensor`): the unit vector directions of the measurements
    Returns: 
        (:obj:`dict`): dictionary with keys and values:

            **lambdas** (:obj:`torch.tensor`): placement of each basis function (frequencies)
  
            **SLambda** (:obj:`torch.tensor`): spectral densities for each basis function (i.e. prior variances)
  
            **Lx,Ly,Lz** (:obj:`torch.tensor`): domain size in x,y,z directions

            **Q1,Q2,Q3,Q4** (:obj:`torch.tensor`): Integral partial results to be used in :func:`basisFuncDInts`.

    """

    [mm1, mm2, mm3] = torch.meshgrid(torch.linspace(1.0,m,m),torch.linspace(1.0,m,m),torch.linspace(1.0,m,m))            # grid of basisfunctions
    insideCircle = torch.sqrt(mm1**2 + mm2**2 + mm3**2) / m <= 1 + 2e-16 # points inside a circle
    mm1 = mm1[insideCircle].double()
    mm2 = mm2[insideCircle].double()
    mm3 = mm3[insideCircle].double()

    dlambda_x = 3.5 / lx / m            # 3.5 sigmacoverage, this coudl be adjusted out to 4.5 sigma
    dlambda_y = 3.5 / ly / m
    dlambda_z = 3.5 / lz / m

    lambda_x = mm1.unsqueeze(0)*dlambda_x        # lambdas
    lambda_y = mm2.unsqueeze(0)*dlambda_y
    lambda_z = mm3.unsqueeze(0)*dlambda_z
    
    Lx = np.pi / 2 / dlambda_x          # Domain scales
    Ly = np.pi / 2 / dlambda_y
    Lz = np.pi / 2 / dlambda_z

    Basis = {'Lx': Lx,
             'Ly': Ly,
             'Lz': Lz}

    if nhat is not None:
        n1 = nhat[[0], :].T
        n2 = nhat[[1], :].T
        n3 = nhat[[2], :].T

        # could calculate this bit in the define basis function part
        n1lx = n1 * lambda_x
        n2ly = n2 * lambda_y
        n3lz = n3 * lambda_z
        Q1 = (n1lx - n2ly - n3lz)
        Q2 = (n1lx + n2ly - n3lz)
        Q3 = (n1lx - n2ly + n3lz)
        Q4 = (n1lx + n2ly + n3lz)

        # look for zeros that will later cause divide by zero errors
        I1 = (Q1.abs() < 1e-10).any(0)
        I2 = (Q2.abs() < 1e-10).any(0)
        I3 = (Q3.abs() < 1e-10).any(0)
        I4 = (Q4.abs() < 1e-10).any(0)

        singular_index = (I1+I2+I3+I4).detach().numpy().astype(np.int)
        max_lam_x = lambda_x.max().detach().numpy()
        max_lam_y = lambda_y.max().detach().numpy()
        max_lam_z = lambda_z.max().detach().numpy()

        # if max_lam is very small the while() will need super many iterations,
        # if max_lam==0 it will fail completely. Best to set a lower bound for
        # these scaling parameters
        if np.abs(max_lam_x)<1.5e-4: max_lam_x = 1.5e-4
        if np.abs(max_lam_y)<1.5e-4: max_lam_y = 1.5e-4
        if np.abs(max_lam_z)<1.5e-4: max_lam_z = 1.5e-4

        maxiteration = 500
        iteration = 0
        while(singular_index.any() and iteration<=maxiteration):
            iteration+=1
            mask = torch.tensor(singular_index)

            lambda_x = lambda_x + mask * 1.5e-6 * max_lam_x * torch.rand(lambda_x.shape)
            lambda_y = lambda_y + mask * 1.5e-6 * max_lam_y * torch.rand(lambda_y.shape)
            lambda_z = lambda_z + mask * 1.5e-6 * max_lam_z * torch.rand(lambda_z.shape)

            n1lx = n1 * lambda_x
            n2ly = n2 * lambda_y
            n3lz = n3 * lambda_z

            Q1 = (n1lx - n2ly - n3lz)
            Q2 = (n1lx + n2ly - n3lz)
            Q3 = (n1lx - n2ly + n3lz)
            Q4 = (n1lx + n2ly + n3lz)

            # look for zeros that will later cause divide by zero errors
            I1 = (Q1.abs() < 1e-10).any(0)
            I2 = (Q2.abs() < 1e-10).any(0)
            I3 = (Q3.abs() < 1e-10).any(0)
            I4 = (Q4.abs() < 1e-10).any(0)
            singular_index = (I1+I2+I3+I4).detach().numpy().astype(np.int)
        assert iteration<maxiteration, 'defineFourierBasis() was unable to remove numerical instability'

        Basis['Q1'] = Q1
        Basis['Q2'] = Q2
        Basis['Q3'] = Q3
        Basis['Q4'] = Q4

    lambdas = torch.cat([lambda_x, lambda_y, lambda_z], dim=0)

    SLambda = sig_f**2 * (2 * np.pi)**(3 / 2) * lx * ly * lz * torch.exp(
        -0.5 * (lambda_x**2 * lx**2 + lambda_y**2 * ly**2 + lambda_z**2 * lz**2))

    Basis['lambdas'] = lambdas
    Basis['SLambda'] = SLambda

    return Basis

def defineBeltramiBasis(m, theta, nhat=None):
    """Defines a set of Fourier basis functions used in the bayesian regression.

    Args:
        m (:obj:`torch.tensor`): Number of basis functions in each frequency dimension.
        theta (:obj:`torch.tensor`): Hyper parameters defining covariance function.
        nhat (:obj:`torch.tensor`): X-ray beam directionin sample coordinate system.

    Returns: 
        (:obj:`tuple` of :obj:`dict`): tuple of dictionaries dictionaries of type :func:`defineFourierBasis`.
        one dictionary for each beltrami tensor componenent. Also returns a concatenated tensor with all 
        dictionaries contained.

    """

    A = defineFourierBasis(m, theta[1], theta[2], theta[3], theta[0], nhat)
    B = defineFourierBasis(m, theta[5], theta[6], theta[7], theta[4], nhat)
    C = defineFourierBasis(m, theta[9], theta[10], theta[11], theta[8], nhat)
    D = defineFourierBasis(m, theta[13], theta[14], theta[15], theta[12], nhat)
    E = defineFourierBasis(m, theta[17], theta[18], theta[19], theta[16], nhat)
    F = defineFourierBasis(m, theta[21], theta[22], theta[23], theta[20], nhat)

    SLambda = torch.cat([A['SLambda'], B['SLambda'], C['SLambda'], D['SLambda'], E['SLambda'], F['SLambda']], 1)

    return A,B,C,D,E,F,SLambda

def basisFuncDerivs(lambdas, Lx, Ly, Lz, x, y, z, req):
    """ Calculates the basis function derivatives Beltrami basis components.

    Args:
        lambdas (:obj:`torch.tensor`): Basis function frequencies.
        Lx,Ly,Lz (:obj:`torch.tensor`): Basis function phases in x,y,z domain.
        x,y,z (:obj:`torch.tensor`): Coordinates at which to calculate basis functions at.
        req (:obj:`list` of :obj:`bool`): Components desired to calculate for.

    Returns: 
        (:obj:`torch.tensor` of :obj:`dict`): Partial derivatives of the basis function, ```shape=6,n,m)```.

    """
    n = x.shape[0]                         # number of points
    mm_adj = lambdas.shape[1]               # total number of basis functions

    # preallocate output
    dfuncs = torch.zeros((n,mm_adj,6)).double()

    # point/measurement stuff always indexes along rows,
    # basis function stuff always indexes along columns,
    Bx = (x + Lx).unsqueeze(1) * lambdas[[0], :]
    By = (y + Ly).unsqueeze(1) * lambdas[[1], :]
    Bz = (z + Lz).unsqueeze(1) * lambdas[[2], :]

    sx = torch.sin(Bx)
    sy = torch.sin(By)
    sz = torch.sin(Bz)
    cx = torch.cos(Bx)
    cy = torch.cos(By)
    cz = torch.cos(Bz)

    lambda_x = lambdas[[0],:]
    lambda_y = lambdas[[1],:]
    lambda_z = lambdas[[2],:]

    if req[0]:   # d/dxdx
        dfuncs[:,:,0] = -(lambda_x**2/torch.sqrt(Lx*Ly*Lz)) * sx * sy * sz

    if req[1]:   # d/dydy
        dfuncs[:,:,1] = -(lambda_y**2/torch.sqrt(Lx*Ly*Lz)) * sx * sy* sz

    if req[2]:   # d/dzdz
        dfuncs[:,:,2] = -(lambda_z**2/torch.sqrt(Lx*Ly*Lz)) * sx * sy * sz

    if req[3]:   # d/dxdy
        dfuncs[:,:,3] = ((lambda_x*lambda_y)/torch.sqrt(Lx*Ly*Lz)) * cx * cy * sz

    if req[4]:   # d/dxdz
        dfuncs[:,:,4] = ((lambda_x*lambda_z)/torch.sqrt(Lx*Ly*Lz)) * cx * sy * cz

    if req[5]:   # d/dydz
        dfuncs[:,:,5] = ((lambda_y*lambda_z)/torch.sqrt(Lx*Ly*Lz)) * sx * cy * cz

    return dfuncs

def BeltramiApproxAnisoStrainField(A,B,C,D,E,F,x,y,z,H):
    """Calculates regressor matrix given hyperparameters for beltrami components and spatial locations.

    This function works for aniostropic materials as the compliance H can be supplied.

    Args:
        A (:obj:`dict`): Basis fuctions for Beltrami component A
        B (:obj:`dict`): Basis fuctions for Beltrami component B
        C (:obj:`dict`): Basis fuctions for Beltrami component C
        D (:obj:`dict`): Basis fuctions for Beltrami component D
        E (:obj:`dict`): Basis fuctions for Beltrami component E
        F (:obj:`dict`): Basis fuctions for Beltrami component F
        x (:obj:`torch.tensor`): x coordinate of points to calculate basis functions at
        y (:obj:`torch.tensor`): y coordinate of points to calculate basis functions at
        z (:obj:`torch.tensor`): z coordinate of points to calculate basis functions at
        H (:obj:`torch.tensor`): the compliance matrix, either a ```shape=(6,6)``` average 
            for the material or ```shape=(N,6,6)``` individual compliances for each point

    Returns: 
        The regressor matrix

    """

    n = len(x)                      # number of test/data points
    mm_adj = A['lambdas'].shape[1]  # total number of basis functions per component

    dfuncs = basisFuncDerivs(A['lambdas'],A['Lx'],A['Ly'],A['Lz'], x, y, z,[0, 1, 1, 0, 0, 1])
    dydyA = dfuncs[:,:,1]
    dzdzA = dfuncs[:,:,2]
    dydzA = dfuncs[:,:,5]

    dfuncs = basisFuncDerivs(B['lambdas'],B['Lx'],B['Ly'],B['Lz'], x, y, z,[1, 0, 1, 0, 1, 0])
    dxdxB = dfuncs[:,:,0]
    dzdzB = dfuncs[:,:,2]
    dxdzB = dfuncs[:,:,4]

    dfuncs = basisFuncDerivs(C['lambdas'],C['Lx'],C['Ly'],C['Lz'], x, y, z,[1, 1, 0, 1, 0, 0])
    dxdxC = dfuncs[:,:,0]
    dydyC = dfuncs[:,:,1]
    dxdyC = dfuncs[:,:,3]

    dfuncs = basisFuncDerivs(D['lambdas'],D['Lx'],D['Ly'],D['Lz'], x, y, z,[0, 0, 1, 1, 1, 1])
    dzdzD = dfuncs[:,:,2]
    dxdyD = dfuncs[:,:,3]
    dxdzD = dfuncs[:,:,4]
    dydzD = dfuncs[:,:,5]

    dfuncs = basisFuncDerivs(E['lambdas'],E['Lx'],E['Ly'],E['Lz'], x, y, z,[0, 1, 0, 1, 1, 1])
    dydyE = dfuncs[:,:,1]
    dxdyE = dfuncs[:,:,3]
    dxdzE = dfuncs[:,:,4]
    dydzE = dfuncs[:,:,5]

    dfuncs = basisFuncDerivs(F['lambdas'],F['Lx'],F['Ly'],F['Lz'], x, y, z,[1, 0, 0, 1, 1, 1])
    dxdxF = dfuncs[:,:,0]
    dxdyF = dfuncs[:,:,3]
    dxdzF = dfuncs[:,:,4]
    dydzF = dfuncs[:,:,5]

    # the basis functions for stress
    phi_sxx = torch.cat([torch.zeros((n, mm_adj)), dzdzB, dydyC, torch.zeros(n, mm_adj), torch.zeros((n, mm_adj)), -2 * dydzF],1)
    phi_syy = torch.cat([dzdzA, torch.zeros((n, mm_adj)), dxdxC, torch.zeros((n, mm_adj)), -2 * dxdzE, torch.zeros((n, mm_adj))],1)
    phi_szz = torch.cat([dydyA, dxdxB, torch.zeros((n, mm_adj)), -2 * dxdyD, torch.zeros((n, mm_adj)), torch.zeros((n, mm_adj))],1)
    phi_sxy = torch.cat([torch.zeros((n, mm_adj)), torch.zeros((n, mm_adj)), -dxdyC, -dzdzD, dydzE, dxdzF],1)
    phi_sxz = torch.cat([torch.zeros((n, mm_adj)), -dxdzB, torch.zeros((n, mm_adj)), dxdzD, dxdyE, -dxdxF],1)
    phi_syz = torch.cat([-dydzA, torch.zeros((n, mm_adj)), torch.zeros((n, mm_adj)), dydzD, -dydyE, dxdyF],1)

    # combined and reshaped to be size [n,6,6*mm_adj]
    phi_s = torch.cat([phi_sxx.unsqueeze(1),phi_syy.unsqueeze(1),phi_szz.unsqueeze(1),phi_sxy.unsqueeze(1),phi_sxz.unsqueeze(1),phi_syz.unsqueeze(1)],1)

    # apply hooke's law to get the basis functions for strain as a [n,6,6*mm_adj]
    phi = torch.matmul(H, phi_s)
    
    # and flatten to be [6*N,6*mm_adj]
    return phi.reshape((6*n,6*mm_adj))

def BeltramiApproxAnisoStressField(A,B,C,D,E,F,x,y,z):
    """Calculates regressor matrix given hyperparameters for beltrami components and spatial locations.

    This function works for aniostropic materials as the compliance H can be supplied.

    Args:
        A (:obj:`dict`): Basis fuctions for Beltrami component A
        B (:obj:`dict`): Basis fuctions for Beltrami component B
        C (:obj:`dict`): Basis fuctions for Beltrami component C
        D (:obj:`dict`): Basis fuctions for Beltrami component D
        E (:obj:`dict`): Basis fuctions for Beltrami component E
        F (:obj:`dict`): Basis fuctions for Beltrami component F
        x (:obj:`torch.tensor`): x coordinate of points to calculate basis functions at
        y (:obj:`torch.tensor`): y coordinate of points to calculate basis functions at
        z (:obj:`torch.tensor`): z coordinate of points to calculate basis functions at

    Returns: 
        The regressor matrix

    NOTE: Approximating stress is simply done by approximating strain with H=I unity.
    """
    I = torch.tensor(np.expand_dims( np.eye(6,6), axis=0)).double()
    return BeltramiApproxAnisoStrainField(A,B,C,D,E,F,x,y,z,H=I)

def basisFuncDInts(lambdas, Lx, Ly, Lz, entry, exit, req, nsegs, Q1, Q2, Q3, Q4):
    """Computes line integrals of the derivatives of basis functions.

    Args:
        lambdas (:obj:`torch.tensor`): placement of each basis function (frequencies)
        Lx,Ly,Lz (:obj:`torch.tensor`): domain size in x,y,z directions
        entry (:obj:`torch.tensor`): entry points of each segment of each measurement ray
        exit (:obj:`torch.tensor`): exit points of each segment of each measurement ray
        req (:obj:`list` of :obj:`bool`): Components desired to calculate for.
        nsegs (:obj:`torch.tensor`): Number of segments for each line integral. ``shape=(N,)``
        Q1,Q2,Q3,Q4 (:obj:`torch.tensor`): Integral partial results computed by :func:`defineFourierBasis`.

    Returns: 
        :obj:`torch.tensor` of ```shape=(n, mm_adj, 6)```, the line integrals of the derivatives of each basis function

    """
    n = entry.shape[1]                      # number of LRTs
    mm_adj = lambdas.shape[1]               # total number of basis functions

    lambda_x = lambdas[[0],:]   # maintain these as row vectors
    lambda_y = lambdas[[1],:]
    lambda_z = lambdas[[2],:]

    # preallocate
    Idfuncs = torch.zeros((n, mm_adj, 6)).double()
    for ss in range(int(nsegs.max().item())):
        segInd = (nsegs > ss).squeeze()      # index of measurements with at least ss+1 segments

        # entry coordinates for this ray segment for each measurement with at least this many segments
        x0 = entry[ss*3,segInd].unsqueeze(1)
        y0 = entry[ss*3+1,segInd].unsqueeze(1)
        z0 = entry[ss*3+2,segInd].unsqueeze(1) # make these column vectors

        alpha_x = (Lx + x0) * lambda_x
        alpha_y = (Ly + y0) * lambda_y
        alpha_z = (Lz + z0) * lambda_z

        Gamma1s = torch.cos(alpha_x - alpha_y - alpha_z)
        Gamma2s = torch.cos(alpha_x + alpha_y - alpha_z)
        Gamma3s = torch.cos(alpha_x - alpha_y + alpha_z)
        Gamma4s = torch.cos(alpha_x + alpha_y + alpha_z)

        # exit coordinates for this ray segment for each measurement with at least this many segments
        xf = exit[ss*3, segInd].unsqueeze(1)
        yf = exit[ss*3+1, segInd].unsqueeze(1)
        zf = exit[ss *3+2, segInd].unsqueeze(1)  # make these column vectors

        alpha_x = (Lx + xf) * lambda_x
        alpha_y = (Ly + yf) * lambda_y
        alpha_z = (Lz + zf) * lambda_z

        Gamma1f = torch.cos(alpha_x - alpha_y - alpha_z)
        Gamma2f = torch.cos(alpha_x + alpha_y - alpha_z)
        Gamma3f = torch.cos(alpha_x - alpha_y + alpha_z)
        Gamma4f = torch.cos(alpha_x + alpha_y + alpha_z)

        Gamma1 = (Gamma1f - Gamma1s) / Q1[segInd,:]    # TODO: how to protect this form having hte wrong shape if segInd is scalar?
        Gamma2 = (Gamma2f - Gamma2s) / Q2[segInd,:]
        Gamma3 = (Gamma3f - Gamma3s) / Q3[segInd,:]
        Gamma4 = (Gamma4f - Gamma4s) / Q4[segInd,:]

        if req[0]: # integral of d / dxdx
            Idfuncs[segInd,:, 0] = Idfuncs[segInd,:, 0] - (lambda_x ** 2 / torch.sqrt(Lx * Ly * Lz) / 4) * (
                    Gamma1 - Gamma2 - Gamma3 + Gamma4)

        if req[1]: # integral of d / dydy
            Idfuncs[segInd,:, 1] = Idfuncs[segInd,:, 1] - (lambda_y ** 2 / torch.sqrt(Lx * Ly * Lz) / 4) * (
                    Gamma1 - Gamma2 - Gamma3 + Gamma4)

        if req[2]: # integral of d / dzdz
            Idfuncs[segInd,:, 2] = Idfuncs[segInd,:, 2] -(lambda_z ** 2 / torch.sqrt(Lx * Ly * Lz) / 4) * (Gamma1-Gamma2-Gamma3+Gamma4)

        if req[3]: # integral of d / dxdy
            Idfuncs[segInd,:, 3] = Idfuncs[segInd,:, 3] +(lambda_x * lambda_y / torch.sqrt(Lx * Ly * Lz) / 4) * (
                    Gamma1 + Gamma2 - Gamma3 - Gamma4)

        if req[4]: # integral of d / dxdz
            Idfuncs[segInd,:, 4] = Idfuncs[segInd,:, 4] +(lambda_x * lambda_z / torch.sqrt(Lx * Ly * Lz) / 4) * (
                    Gamma1 - Gamma2 + Gamma3 - Gamma4)

        if req[5]: # integral of d / dydz
            Idfuncs[segInd,:, 5] = Idfuncs[segInd,:, 5] +(lambda_y * lambda_z / torch.sqrt(Lx * Ly * Lz) / 4) * (
                    -Gamma1 - Gamma2 - Gamma3 - Gamma4)
    return Idfuncs

def BeltramiApproxAnisoRayMeas(A,B,C,D,E,F,H,entry,exit,nsegs,L, kappa,subset_idx=None):
    """
    """
    n = entry.shape[1]                      # number of ray measurements
    mm_adj = A['lambdas'].shape[1]  # total number of basis functions per component
    idx = subset_idx

    if idx is None:
        Idfuncs = basisFuncDInts(A['lambdas'], A['Lx'], A['Ly'], A['Lz'], entry, exit, [0, 1, 1, 0, 0, 1], nsegs,
                       A['Q1'], A['Q2'], A['Q3'], A['Q4'])
    else:
        Idfuncs = basisFuncDInts(A['lambdas'], A['Lx'], A['Ly'], A['Lz'], entry, exit, [0, 1, 1, 0, 0, 1], nsegs,
                       A['Q1'][idx,:], A['Q2'][idx,:], A['Q3'][idx,:], A['Q4'][idx,:])
    IdydyA = Idfuncs[:,:,1]
    IdzdzA = Idfuncs[:,:,2]
    IdydzA = Idfuncs[:,:,5]

    if idx is None:
        Idfuncs = basisFuncDInts(B['lambdas'], B['Lx'], B['Ly'], B['Lz'], entry, exit, [1, 0, 1, 0, 1, 0], nsegs,
                   B['Q1'], B['Q2'], B['Q3'], B['Q4'])
    else:
        Idfuncs = basisFuncDInts(B['lambdas'], B['Lx'], B['Ly'], B['Lz'], entry, exit, [1, 0, 1, 0, 1, 0], nsegs,
                   B['Q1'][idx,:], B['Q2'][idx,:], B['Q3'][idx,:], B['Q4'][idx,:])
    IdxdxB = Idfuncs[:,:,0]
    IdzdzB = Idfuncs[:,:,2]
    IdxdzB = Idfuncs[:,:,4]

    if idx is None:
        Idfuncs = basisFuncDInts(C['lambdas'], C['Lx'], C['Ly'], C['Lz'], entry, exit, [1, 1, 0, 1, 0, 0], nsegs,
                   C['Q1'], C['Q2'], C['Q3'], C['Q4'])
    else:
        Idfuncs = basisFuncDInts(C['lambdas'], C['Lx'], C['Ly'], C['Lz'], entry, exit, [1, 1, 0, 1, 0, 0], nsegs,
                   C['Q1'][idx,:], C['Q2'][idx,:], C['Q3'][idx,:], C['Q4'][idx,:])
    IdxdxC = Idfuncs[:,:,0]
    IdydyC = Idfuncs[:,:,1]
    IdxdyC = Idfuncs[:,:,3]

    if idx is None:
        Idfuncs = basisFuncDInts(D['lambdas'], D['Lx'], D['Ly'], D['Lz'], entry, exit, [0, 0, 1, 1, 1, 1], nsegs,
                   D['Q1'], D['Q2'], D['Q3'], D['Q4'])
    else:
        Idfuncs = basisFuncDInts(D['lambdas'], D['Lx'], D['Ly'], D['Lz'], entry, exit, [0, 0, 1, 1, 1, 1], nsegs,
                   D['Q1'][idx,:], D['Q2'][idx,:], D['Q3'][idx,:], D['Q4'][idx,:])
    IdzdzD = Idfuncs[:,:,2]
    IdxdyD = Idfuncs[:,:,3]
    IdxdzD = Idfuncs[:,:,4]
    IdydzD = Idfuncs[:,:,5]

    if idx is None:
        Idfuncs = basisFuncDInts(E['lambdas'], E['Lx'], E['Ly'], E['Lz'], entry, exit, [0, 1, 0, 1, 1, 1], nsegs,
                   E['Q1'], E['Q2'], E['Q3'], E['Q4'])
    else:
        Idfuncs = basisFuncDInts(E['lambdas'], E['Lx'], E['Ly'], E['Lz'], entry, exit, [0, 1, 0, 1, 1, 1], nsegs,
                   E['Q1'][idx,:], E['Q2'][idx,:], E['Q3'][idx,:], E['Q4'][idx,:])
    IdydyE = Idfuncs[:,:,1]
    IdxdyE = Idfuncs[:,:,3]
    IdxdzE = Idfuncs[:,:,4]
    IdydzE = Idfuncs[:,:,5]

    if idx is None:
        Idfuncs = basisFuncDInts(F['lambdas'], F['Lx'], F['Ly'], F['Lz'], entry, exit, [1, 0, 0, 1, 1, 1], nsegs,
                   F['Q1'], F['Q2'], F['Q3'], F['Q4'])
    else:
        Idfuncs = basisFuncDInts(F['lambdas'], F['Lx'], F['Ly'], F['Lz'], entry, exit, [1, 0, 0, 1, 1, 1], nsegs,
                   F['Q1'][idx,:], F['Q2'][idx,:], F['Q3'][idx,:], F['Q4'][idx,:])
    IdxdxF = Idfuncs[:,:,0]
    IdxdyF = Idfuncs[:,:,3]
    IdxdzF = Idfuncs[:,:,4]
    IdydzF = Idfuncs[:,:,5]

    K1K1 = kappa[[0],:].T**2    # ensuring things to do with measurements are column vectors
    K2K2 = kappa[[1],:].T**2
    K3K3 = kappa[[2],:].T**2
    K1K2 = 2 * kappa[[0],:].T * kappa[[1],:].T
    K1K3 = 2 * kappa[[0], :].T * kappa[[2], :].T
    K2K3 = 2 * kappa[[1], :].T * kappa[[2], :].T

    # integrals of the basis functions for stress
    phi_Isxx = torch.cat([torch.zeros((n, mm_adj)), IdzdzB, IdydyC, torch.zeros(n, mm_adj), torch.zeros((n, mm_adj)), -2 * IdydzF],1)
    phi_Isyy = torch.cat([IdzdzA, torch.zeros((n, mm_adj)), IdxdxC, torch.zeros((n, mm_adj)), -2 * IdxdzE, torch.zeros((n, mm_adj))],1)
    phi_Iszz = torch.cat([IdydyA, IdxdxB, torch.zeros((n, mm_adj)), -2 * IdxdyD, torch.zeros((n, mm_adj)), torch.zeros((n, mm_adj))],1)
    phi_Isxy = torch.cat([torch.zeros((n, mm_adj)), torch.zeros((n, mm_adj)), -IdxdyC, -IdzdzD, IdydzE, IdxdzF],1)
    phi_Isxz = torch.cat([torch.zeros((n, mm_adj)), -IdxdzB, torch.zeros((n, mm_adj)), IdxdzD, IdxdyE, -IdxdxF],1)
    phi_Isyz = torch.cat([-IdydzA, torch.zeros((n, mm_adj)), torch.zeros((n, mm_adj)), IdydzD, -IdydyE, IdxdyF],1)

    # combined and reshaped to be size [n,6,6*mm_adj]
    phi_Is = torch.cat([phi_Isxx.unsqueeze(1),phi_Isyy.unsqueeze(1),phi_Iszz.unsqueeze(1),phi_Isxy.unsqueeze(1),
                        phi_Isxz.unsqueeze(1),phi_Isyz.unsqueeze(1)],1)

    # converted to integrals of the strain basis functions by applying hooke's law
    phi_Istrain = torch.matmul(H,phi_Is)

    # converted to the basis function of the measurements by applying \bar{\kappa}
    Phi = K1K1 * phi_Istrain[:,0,:] + K2K2 * phi_Istrain[:,1,:] + K3K3 * phi_Istrain[:,2,:] + \
          K1K2 * phi_Istrain[:,3,:] + K1K3 * phi_Istrain[:,4,:] + K2K3 * phi_Istrain[:,5,:]

    return Phi / L.view(n, 1)  # make sure L is column vec

def negativeLogLikelihood(Phi, Y, sig_m, v, r):
    """Computes the negative log likelihood based on gaussian measurement model.

    cost =  0.5*error^T Sigma^{-1} error + 0.5 log(det(Sigma)) + C
    where Sigma is the predictions of the measurement variances
    Sigma = Variance of the predicted value + variance of the measurements (sig_m^2)
    This computation is done in square root form for computational efficiency and
    numerical accuracy

    Args:
        Phi (:obj:`torch.tensor`): the regressor matrix
        Y (:obj:`torch.tensor`): the measurement values
        sig_m (:obj:`torch.tensor`): the measurement standard deviations, either a scalar or an N,1 vector of individual stds
        v (:obj:`torch.tensor`): the fit basis function coefficients
        r (:obj:`torch.tensor`): the cholesky factor calculated during regression

    Returns:
        The negative log likelihood

    """
    predictions, chol_var = predict(Phi, v, r)
    n = len(predictions)
    Gamma = torch.cat([chol_var, torch.diag(sig_m.squeeze()*torch.ones(n,))], 0)
    (_, r) = torch.qr(Gamma)
    error = Y - predictions
    (alpha, _) = torch.triangular_solve(error.unsqueeze(1), r, transpose=True)
    nll = 0.5 * torch.dot(alpha.squeeze(),alpha.squeeze()) + r.diag().abs().log().sum()
    return nll


def nll_scipy_wrapper_RT_aniso(logtheta, Y_train, entry_train, exit_train, nsegs_train, L_train, kappa_train,
                          Y_test, entry_test,exit_test,nsegs_test,L_test, kappa_test, sig_m_train, sig_m_test,
                         Hray_train, Hray_test, m, nhat, train_idx, test_idx):
    """Computes the negative log likelihood based on gaussian measurement model.
    """
    logtheta = torch.tensor(logtheta, requires_grad=True).double()
    theta = logtheta.exp()
    A, B, C, D, E, F, SLambda = defineBeltramiBasis(m, theta[:24], nhat=nhat)
    Phi_train = BeltramiApproxAnisoRayMeas(A, B, C, D, E, F, Hray_train, entry_train,
                                      exit_train, nsegs_train, L_train, kappa_train, subset_idx=train_idx)
    v, r = regression(Phi_train, Y_train, sig_m_train, SLambda)
    Phi_test = BeltramiApproxAnisoRayMeas(A, B, C, D, E, F, Hray_test, entry_test,
                                     exit_test, nsegs_test, L_test, kappa_test, subset_idx=test_idx)
    loss = negativeLogLikelihood(Phi_test, Y_test, sig_m_test, v, r)
    loss.backward()
    grad = logtheta.grad
    return loss.detach().numpy(), grad.detach().numpy()

def regression(Phi, Y, sig_m, SLambda):
    """Performs linear bayesian regression to fit basis function coefficients.

    Args:
        Phi (:obj:`torch.tensor`): The regressor matrix
        Y (:obj:`torch.tensor`): The measurements
        sig_m (:obj:`torch.tensor`): The measurement standard deviation, either a scalar or a N,1 array of individual stds
        SLambda (:obj:`torch.tensor`): The prior covariances for the basis functions

    Returns:
        (v,r) where v is the basis function coefficients and r is the cholesky factor required form stuffs

    """
    # n, m = Phi.shape
    Gamma = torch.cat([Phi / sig_m, torch.diag(1 / SLambda.sqrt().squeeze())], 0)
    (_, r) = torch.qr(Gamma)  # does a thing QR decomp
    (tmp, _) = torch.triangular_solve(Phi.T.matmul(Y.unsqueeze(1) / sig_m ** 2), r, transpose=True)
    (v, _) = torch.triangular_solve(tmp, r, transpose=False)
    return v, r

def predict(Phi, v, r):
    """Predicts the point location values of a field given regressor matrix.
    
    Based on the fitted basis function coefficients v, and the cholesky factor r
    also returns the cholesky factor of the variances of these predictions

    Args:
        Phi (:obj:`torch.tensor`): The regressor matrix
        v (:obj:`torch.tensor`): The fit basis function coefficients
        r (:obj:`torch.tensor`): The cholesky factor calculated during regression

    Returns:
        predictions, chol_var
    """
    predictions = Phi.matmul(v).squeeze()
    (chol_var, _) = torch.triangular_solve(Phi.T, r, transpose=True)
    return predictions, chol_var