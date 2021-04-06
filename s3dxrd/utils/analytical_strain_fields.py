import numpy as np
from scipy.interpolate import griddata,interp2d
import matplotlib.pyplot as plt
from xfab import tools

'''Module for defining various analytical strain fields
'''

def cantilevered_beam( t, h, l, E, nu, Py, Iyy, Pz, Izz, tensor=True ):
    '''Defines strain in a cantilevered beam gicen some material parameters
    '''
    
    # strain field
    Fxx = lambda x, y, z: Py*(l-x)*y/E/Iyy + Pz*(l-x)*z/E/Izz
    Fyy = lambda x, y, z: -nu*Py*(l-x)*y/E/Iyy + -nu*Pz*(l-x)*z/E/Izz
    Fzz = lambda x, y, z: -nu*Py*(l-x)*y/E/Iyy + -nu*Pz*(l-x)*z/E/Izz
    Fxy = lambda x, y, z: -(1+nu)/E*Py*(h**2/4-y**2)/2/Iyy
    Fxz = lambda x, y, z: -(1+nu)/E*Pz*(t**2/4-z**2)/2/Izz
    Fyz = lambda x, y, z: 0*x + 0*y +0*z

    if tensor:
        # input vector x get tensor strain
        strain = lambda x: np.array( [
                        [Fxx(x[0],x[1],x[2]),Fxy(x[0],x[1],x[2]),Fxz(x[0],x[1],x[2])],
                        [Fxy(x[0],x[1],x[2]),Fyy(x[0],x[1],x[2]),Fyz(x[0],x[1],x[2])],
                        [Fxz(x[0],x[1],x[2]),Fyz(x[0],x[1],x[2]),Fzz(x[0],x[1],x[2])]
                        ] )
        return strain
    else:
        # input scalar x,y,z get scalars Fxx, Fyy, Fzz, Fxy, Fxz, Fyz
        return Fxx, Fyy, Fzz, Fxy, Fxz, Fyz

def maxwell_linear_strain(self, compliance, a, b, c, d, tx, ty, tz):
    '''Produce a callable function as fun(x,y,z) <-- strain tensor 
    at coordinate x,y,z, such that the strain field is in equlibrium 
    given a specific linear elastic compliance matrix using the
    Maxwell stress functions 
        PHI_ij = [A, 0, 0]
                 [0, B, 0]
                 [0, 0, C]
    and setting
        A = B = C = f(x,y,z)
        f(x,y,z = a*(x-cx)**3 + b*(y-cy)**3 + c*(z-cx)**3 + d*(x-cx)*(y-cy)*(z-cz)
    this will lead to the strain being linear in the spatial coordinates x,y,z.
    '''

    d2A_dydy = lambda x,y,z: b * ( 6*(y-ty) )
    d2A_dzdz = lambda x,y,z: c * ( 6*(z-tz) )
    d2A_dydz = lambda x,y,z: d * ( (x-tx) )

    d2B_dxdx = lambda x,y,z: a * ( 6*(x-tx) )
    d2B_dzdz = lambda x,y,z: c * ( 6*(z-tz) )
    d2B_dzdx = lambda x,y,z: d * ( (y-ty) )

    d2C_dxdx = lambda x,y,z: a * ( 6*(x-tx) )
    d2C_dydy = lambda x,y,z: b * ( 6*(y-ty) )
    d2C_dxdy = lambda x,y,z: d * ( (z-tz) )

    sigma_xx = lambda x,y,z:  d2B_dzdz(x,y,z) + d2C_dydy(x,y,z)
    sigma_yy = lambda x,y,z:  d2C_dxdx(x,y,z) + d2A_dzdz(x,y,z)
    sigma_zz = lambda x,y,z:  d2A_dydy(x,y,z) + d2B_dxdx(x,y,z)
    sigma_xy = lambda x,y,z: -d2C_dxdy(x,y,z)
    sigma_xz = lambda x,y,z: -d2B_dzdx(x,y,z)
    sigma_yz = lambda x,y,z: -d2A_dydz(x,y,z)

    sigma = lambda x,y,z: np.array([ sigma_xx(x,y,z) , sigma_yy(x,y,z) , sigma_zz(x,y,z) , \
                                     sigma_xy(x,y,z) , sigma_xz(x,y,z) , sigma_yz(x,y,z)  ])

    def strain(x,y,z):
        eps_vec = compliance.dot( sigma(x,y,z) )
        eps = np.array([[ eps_vec[0]   ,  eps_vec[3]/2 , eps_vec[4]/2  ],
                        [ eps_vec[3]/2 ,  eps_vec[1]   , eps_vec[5]/2  ],
                        [ eps_vec[4]/2 ,  eps_vec[5]/2 , eps_vec[2]    ]])
        return eps

    return strain

def field_from_simulation( paths, zpos ):
    '''Read the strain field from a list of simulation files, and return a 
    callable function that returns interpolated strain at specified X,Y,Z-
    grid coordinates. If only X and Y is passed, the read data in paths is 
    assumed to be 2D.
    '''
    coordinates, strains, euler_angles = [],[],[]
    
    for z,path in zip(zpos, paths):
        voxel_id = 0
        U = np.zeros((3,3))
        with open(path) as f:
            inp = f.readlines()
            for i,line in enumerate(inp):
                if 'pos_voxels_'+str(voxel_id) in line:
                    coordinates.append( [ float(line.split()[1]), float(line.split()[2]), z/float(1e3) ] )
                if 'U_voxels_'+str(voxel_id) in line:
                    U[0,0], U[0,1], U[0,2], U[1,0], U[1,1], U[1,2], U[2,0], U[2,1], U[2,2] = [float(number) for number in line.split()[1:]]
                if 'eps_voxels_'+str(voxel_id) in line:
                    eps11, eps12, eps13, eps22, eps23, eps33 = [float(number) for number in line.split()[1:]]
                    strain_tensor = np.array([[eps11,eps12,eps13],
                                              [eps12,eps22,eps23],
                                              [eps13,eps23,eps33]])
                    strain_tensor = ( U.dot( strain_tensor ) ).dot( U.T ) # to sample system
                    strains.append( [ strain_tensor[0,0], strain_tensor[1,1], strain_tensor[2,2], strain_tensor[0,1], strain_tensor[0,2],  strain_tensor[1,2] ] )
                    euler1, euler2, euler3 =  tools.u_to_euler( U )
                    euler_angles.append([euler1, euler2, euler3])
                    voxel_id+=1
    coordinates, strains = np.array(coordinates)*1e3, np.array(strains) # mm scaled to microns
    euler_angles = np.degrees( np.array( euler_angles ) )

    def strain_function( X, Y, Z=None ):
        
        if Z is None:
            points = coordinates[:,0:2]
            xi = np.array([ X.flatten(), Y.flatten() ]).T
        else:
            points = coordinates
            xi = np.array([ X.flatten(), Y.flatten(), Z.flatten() ]).T

        interp_strain = []
        for i in range(6):
            s = griddata(points, strains[:,i], xi, method='nearest')
            if Z is None:
                s = s.reshape(X.shape[0], X.shape[1])
            else:
                s = s.reshape(X.shape[0], X.shape[1], X.shape[2])
            interp_strain.append( s )
        return interp_strain

    def euler_function( X, Y, Z=None ):
        
        if Z is None:
            points = coordinates[:,0:2]
            xi = np.array([ X.flatten(), Y.flatten() ]).T
        else:
            points = coordinates
            xi = np.array([ X.flatten(), Y.flatten(), Z.flatten() ]).T

        interp_euler = []
        for i in range(3):
            e = griddata(points, euler_angles[:,i], xi, method='nearest')
            if Z is None:
                e = e.reshape(X.shape[0], X.shape[1])
            else:
                e = e.reshape(X.shape[0], X.shape[1], X.shape[2])
            interp_euler.append( e )
        return interp_euler

    return strain_function, euler_function, coordinates


def discrete_to_analytic( strains, xvec, yvec ):
    ''' Get an analytical function for evaluating strain at points in a plane. 
    strain in format:
        strain=[ XX , YY, ZZ , XY , XZ , YZ]
    and X,Y,Z are meshgrids
    '''
    eps_xx_func = interp2d(xvec, yvec, strains[0],  kind='linear')
    eps_yy_func = interp2d(xvec, yvec, strains[1],  kind='linear')
    eps_zz_func = interp2d(xvec, yvec, strains[2],  kind='linear')
    eps_xy_func = interp2d(xvec, yvec, strains[3],  kind='linear')
    eps_xz_func = interp2d(xvec, yvec, strains[4],  kind='linear')
    eps_yz_func = interp2d(xvec, yvec, strains[5],  kind='linear')
    def interp_strain( x ):
        eps_xx = eps_xx_func(x[0],x[1])[0]
        eps_yy = eps_yy_func(x[0],x[1])[0]
        eps_zz = eps_zz_func(x[0],x[1])[0]
        eps_xy = eps_xy_func(x[0],x[1])[0]
        eps_xz = eps_xz_func(x[0],x[1])[0]
        eps_yz = eps_yz_func(x[0],x[1])[0]
        eps = np.array([[eps_xx, eps_xy, eps_xz],
                        [eps_xy, eps_yy, eps_yz],
                        [eps_xz, eps_yz, eps_zz]])
        return eps
    return interp_strain




