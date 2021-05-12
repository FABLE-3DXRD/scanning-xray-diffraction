import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon as shapelyPolygon
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, diags
from scipy.optimize import LinearConstraint

def _get_A_matrix_row( centroids, polymesh, n, N, entry, exit, nhat, beam_width ):
    
    zhat = np.array([0, 0, 1])
    that = np.cross( nhat, zhat )
    a = entry[0:3]  - nhat*len(polymesh)*beam_width + that*(beam_width/2.)
    b = entry[0:3]  - nhat*len(polymesh)*beam_width - that*(beam_width/2.)
    c = entry[0:3]  + nhat*len(polymesh)*beam_width - that*(beam_width/2.)
    d = entry[0:3]  + nhat*len(polymesh)*beam_width + that*(beam_width/2.)
    polybeam = shapelyPolygon([(a[0], a[1]), (b[0], b[1]), (c[0], c[1]), (d[0], d[1])])
    
    ts    = nhat[0:2].dot( (centroids - entry[0:2]).T )
    nn = nhat[0:2].reshape(2,1)
    en = entry[0:2].reshape(2,)
    dist = np.linalg.norm( en + (ts*nn).T - centroids, axis=1 )
    mask = dist < beam_width*(1+np.sqrt(2))/2.
    areas = _compute_intersection_areas( polymesh, polybeam, mask, plot=False )

    if np.sum(areas)==0: 
        return None
    else:
        return np.repeat(areas,6)*_directional_weights(n, N)/np.sum(areas)

def _compute_intersection_areas( polymesh, polybeam, mask, plot=False ):

    intersection_areas = np.zeros(len(polymesh))

    for i,poly_elm in enumerate(polymesh):

        if mask[i]:
            intersection_areas[i] = polybeam.intersection(poly_elm).area
        
        if plot:
            print("mask[i]",mask[i])
            plt.plot(polybeam.exterior.xy[0],polybeam.exterior.xy[1])
            for m in polymesh:
                plt.plot(m.exterior.xy[0],m.exterior.xy[1],"k")
            plt.plot(poly_elm.exterior.xy[0],poly_elm.exterior.xy[1],"r")
            plt.title(intersection_areas[i])
            plt.show()

    return intersection_areas

def _directional_weights( n, N ):
    """
    Compute the directional weights of a particular w:y scan beam
    (to be used to form a row in the A matrix)
    """
    return np.array( [ n[0]*n[0], n[1]*n[1], n[2]*n[2], 2*n[1]*n[2], 2*n[0]*n[2], 2*n[0]*n[1] ]*N )


def _calc_A_matrix( mesh, directions, entry, exit, nhat, beam_width, verbose  ):
    M = directions.shape[1] # number of measurements
    N = mesh.shape[0] # number of elements

    row_indx, col_indx, data = [],[],[]
    polymesh = [shapelyPolygon([(e[0], e[1]), (e[2], e[3]), (e[4], e[5]), (e[6], e[7])]) for e in mesh]
    centroids = np.array( [ np.mean(mesh[:,0::2],axis=1), np.mean(mesh[:,1::2],axis=1)] ).T

    bad_equations = []
    count=0
    if verbose:
        print("Building projection matrix...")
    for k in range(M):
        row = _get_A_matrix_row( centroids, polymesh, directions[:,k], N, entry[:,k], exit[:,k], nhat[:,k], beam_width )
        if k%300==0:
            if verbose:
                print("Getting row number: "+str(k) +" of "+str(M))
        if row is None:
            bad_equations.append(k)
            count+=1
        else:
            ci = list(np.where(row!=0)[0])
            col_indx.extend( ci )
            row_indx.extend( [k]*len(ci) )
            data.extend( row[ci] )
    if verbose:
        print("")
        print("Total number of eqs: "+str( M) )
        print("Topology cutoff lead to "+str( 100.*count/float(M) )+" percent eqs to be unusuable")
        print("")

    return csr_matrix((data, (row_indx, col_indx)), shape=(M,N*6)), bad_equations

def _constraints(mesh, low_bound, high_bound):
    """
    Limit the difference in strain between to neighbouring elements
    A neighbour pair is defined as two elements sharing at least one node
    """
    c = []
    incl = []

    data = []
    row = []
    col = []
    curr_row = 0
    for i,elm in enumerate(mesh):
        indx = _find_index_of_neighbors(mesh, elm)
        for j in indx:
            if [i,j] in incl or [j,i] in incl: continue
            
            for k in range(6):

                row.append( curr_row )
                row.append( curr_row )
                curr_row+=1
                data.append(1.)
                col.append( (i*6)+k)
                data.append(-1.)
                col.append((j*6)+k)

            incl.append([i,j])
    r,c = curr_row, mesh.shape[0]*6

    c = csr_matrix( (data, (row, col)), shape=(r, c) )
    lb = np.ones(c.shape[0])*low_bound
    ub = np.ones(c.shape[0])*high_bound

    return lb, c, ub


def _find_index_of_neighbors(mesh, element):
    elm_side = np.max(element[0::2]) - np.min(element[0::2])
    index_neighbors = []
    for i,elm in enumerate(mesh):
        if sum(elm==element)==len(elm): continue
        is_neighbour = False
        for x,y in zip(elm[0::2],elm[1::2]):
            for x_e,y_e in zip(element[0::2],element[1::2]):
                if abs(x-x_e)<elm_side/10. and abs(y-y_e)<elm_side/10.:
                    is_neighbour = True
        if is_neighbour: index_neighbors.append(i)
    return np.array( index_neighbors )


def trust_constr_solve( mesh, 
                        directions, 
                        strains , 
                        entry, 
                        exit,
                        nhat,
                        weights, 
                        beam_width, 
                        grad_constraint, 
                        maxiter,
                        verbose=True ):
    """Compute a voxelated strain-tensor field weighted least squares fit to a series of line integral strain measures.
    
    Assigns strain tensors over a 2d input mesh on a per element basis.

    Args:
        mesh (:obj:`numpy array`): Coordinate of mesh element nodes, ```shape=(N,4)``` for quads.
        directions (:obj:`numpy array`): unit vectors along which :obj:`strains` apply.
        strains (:obj:`numpy array`): Average strains along lines.
        entry (:obj:`numpy array`): Xray grian entry points per line.
        exit (:obj:`numpy array`): Xray grian exit points per line.
        nhat (:obj:`numpy array`): Xray direction per integral in sample coordinate system.
        weights (:obj:`numpy array`): Per measurement weights, higher weight gives more impact of equation.
        beam_width (:obj:`float`): Width of beam in units of microns.
        grad_constraint (:obj:`float`): In x-y plane smoothing constraint of strain reconstruction. Deviations between 
            two neighbouring pixels cannot exceed this value for any strain component.
        maxiter (:obj:`int`): Maximum number of WLSQ iterations to perform.
        verbose (:obj:`bool`): If to print progress. Defaults to True.

    Returns:
        (:obj:`list` of :obj:`numpy array`): List of strain tensor components each with ```shape=mesh.shape```. 
        The order of components is "XX","YY","ZZ","YZ","XZ","XY".

    """

    nelm = mesh.shape[0]

    A, bad_equations = _calc_A_matrix( mesh, directions, entry, exit, nhat, beam_width, verbose )

    mask = np.ones(A.shape[0], dtype=bool)
    mask[bad_equations] = False
    A = A[mask]

    strains = np.delete(strains, bad_equations, axis=0)
    weights = np.delete(weights, bad_equations, axis=0)

    if verbose:
        print("Computing _constraints matrix")
    lb,c,ub = _constraints(mesh, -grad_constraint, grad_constraint)

    linear_constraint = LinearConstraint(c, lb, ub, keep_feasible=True)
    x0 = np.zeros(6*nelm)

    def callback( xk, state ):
        if verbose:
            out="   {}      {}      {}"
            if state.nit==1: print(out.format("iteration","cost","max strain grad") )
            print( out.format( state.nit, np.round(state.fun,9), np.max(np.abs(c.dot(xk))) ) )
        return state.nit==maxiter

    W = diags(weights)
    m = strains

    def func( x ): return 0.5*np.linalg.norm( (W.dot( A.dot(x) ) - W.dot( m )) )**2
    def jac( x ): return A.T.dot( W.T.dot( W.dot( A.dot(x) ) - W.dot( m ) ) )
    
    res = minimize(func, x0, method="trust-constr", jac=jac, \
                    callback=callback, tol=1e-8, \
                    constraints=[linear_constraint],\
                    options={"disp": verbose, "maxiter":maxiter})

    s_tilde = res.x

    # reformat, each row is strain for the element
    s_tilde = s_tilde.reshape(nelm,6)

    return [s_tilde[:,0],s_tilde[:,1],s_tilde[:,2],s_tilde[:,3],s_tilde[:,4],s_tilde[:,5]]