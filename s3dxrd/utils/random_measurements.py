import numpy as np
import shapely.geometry
from shapely import affinity
from scipy.integrate import quad
import matplotlib.pyplot as plt

def simulate_line_measurements( angs, strain, ytrans, ztrans, sample_corners, show_geom=False ):
    '''Simulate parallel ray integrals over a strain field in a 3D sample by looping 
       over z-positions. Convention here is that coordinate x is along the xray for an
       angle of zero degrees. The sample is represented as a stack of 2D polygons.

    Input:
        angs            : array of angular settings in degrees of the sample
        strain          : function that takes a coordinate array x=(x,y,z) and returns a numpy (3x3) strain tensor 
        ytrans          : sample y coordinate translation for measurements
        ztrans          : sample z coordinate translation for measurements
        sample_corners  : List containing the sample corners for each ztrans
        show_geom       : debug option, plots the sample and xray for each measurement

    Returns:
        Y      : is N by 1 array of the measurement values
        sig_m  : is N by 1 array of the standard deviations of the measurement values
        entry  : is a 3 by N array of the coordinates where the ray enters the sample (each col is x, y, z)
        exit   : is a 3 by N array of the coordinates where the ray leaves the sample (each col is x, y,z)
        nhat   : is a 3 by N array of directions of each ray (each col is a unit vector)
        kappa  : is a 3 by N array that defines the direction of strain measured (each col is a unit vector)
        L      : is a 1 by N array of the total irradiated lengths of each ray (ie distance between entry and exit)
        nsegs  : is a 1 by N array of all ones (number of segments for each ray) this is useful when we go to non convex geometry

        (If we have non convex geometry then
        nsegs is a 1 by N array containing the number of times that the ray enters and exits the sample
        entry becomes a p*3 by N array where p is max(nsegs) and each col is teh stacked entries so [first entry x, first entry y, first entry z, second entry x, second entry y, and so on]
        exit becomes a p*3 by N array with a similar story)

    NOTE: If a ray is outside the sample all values in the output arrays are put to zero for this 
          measurement setting. The same is true if the ray only graces a corner point.
    '''

    # Instantiate lists to contain all measurements
    all_Y, all_sig_m, all_entry, all_exit, all_nhat, all_kappa, all_L, all_nsegs = [],[],[],[],[],[],[],[]
    xray_endpoints = get_xray_endpoints( sample_corners )

    # Loop over all experimental settings
    for ang in angs:

        # select a random kappa for this angle
        strain_dir = np.random.rand( 3 )
        strain_dir = strain_dir/np.linalg.norm(strain_dir)

        for z,sample_bounds in zip(ztrans, sample_corners):

            # extract the sample geometry at this z-coordinate
            sample_polygon = shapely.geometry.Polygon(sample_bounds)

            for y in ytrans:

                # Translate and rotate the xray endpoints according to ytrans and angle
                c, s = np.cos( np.radians( -ang ) ), np.sin( np.radians( -ang ) )
                rotz = np.array([[c,-s],[s,c]])
                rx = rotz.dot( xray_endpoints + np.array([[0,0],[-y,-y]]) )
                xray_polygon = shapely.geometry.LineString( [ rx[:,0], rx[:,1] ] )
                
                if show_geom:
                    # Show sample and xray polygons
                    xc,yc = xray_polygon.xy
                    plt.plot(xc,yc)
                    xc,yc = sample_polygon.exterior.xy
                    plt.plot(xc,yc)
                    plt.title(' angle='+str(ang)+r'$^o$,   y_translation='+str(y)+',   z_translation='+str(z))
                    plt.axis('equal')
                    plt.show()

                # compute the intersections between beam and sample
                intersection_points = get_intersection( xray_polygon, sample_polygon, z  )
                
                if intersection_points is None:
                    # If a measurement missed the sample or graced a corner, we skipp ahead
                    continue
                else:
                    # make a measurement at the current setting
                    Y,sig_m,entry,exit,nhat,kappa,L,nsegs = measure( strain, intersection_points, strain_dir )
                
                    # save the measurement results in global lists
                    all_Y.append( Y )
                    all_sig_m.append( sig_m )
                    all_entry.append( entry )
                    all_exit.append( exit )
                    all_nhat.append( nhat )
                    all_kappa.append( kappa )
                    all_L.append( L )
                    all_nsegs.append( nsegs )

    # repack lists of measurements into numpy arrays of desired format
    Y, sig_m, entry, exit, nhat, kappa, L, nsegs = repack(all_Y, all_sig_m, all_entry, all_exit, all_nhat, all_kappa, all_L, all_nsegs)

    return Y, sig_m, entry, exit, nhat, kappa, L, nsegs


def get_xray_endpoints( sample_xy_corners ):
    '''Calculate endpoitns of xray line segement. The lenght of the 
    line segment is adapted to make sure xray always convers the full
    length of the sample.
    '''
    xmin = np.min( [ np.min( [c[0] for c in sc] ) for sc in sample_xy_corners ] )
    xmax = np.max( [ np.max( [c[0] for c in sc] ) for sc in sample_xy_corners ] )
    ymin = np.min( [ np.min( [c[1] for c in sc] ) for sc in sample_xy_corners ] )
    ymax = np.max( [ np.max( [c[1] for c in sc] ) for sc in sample_xy_corners ] )
    D = np.sqrt( (xmax-xmin)**2 + (ymax-ymin)**2 )
    return np.array([ [-1.1*D, 1.1*D], [0,0] ])                      


def get_intersection(xray_polygon, sample_polygon, z ):
    '''Compute the 3d coordinates of intersection between xray and
    sample.
    '''
    intersection = sample_polygon.intersection( xray_polygon )
    if intersection.is_empty:
        # we missed the sample with the beam
        intersection_points = None
    elif isinstance(intersection, shapely.geometry.linestring.LineString):
        # we got a single line segment intersection
        intersection_points = np.zeros( (2,3) )
        intersection_points[:2,:2] = np.array( intersection.xy ).T
        intersection_points[:,2] = -z
    elif isinstance(intersection, shapely.geometry.multilinestring.MultiLineString):
        # we got multiple line segments intersection
        intersection_points = np.zeros( (2*len(intersection.geoms),3) )
        for i,line_segment in enumerate(intersection.geoms):
            intersection_points[2*i:2*(i+1),:2] = np.array( line_segment.xy ).T
        intersection_points[:,2] = -z
    
    return intersection_points


def repack(all_Y, all_sig_m, all_entry, all_exit, all_nhat, all_kappa, all_L, all_nsegs):
    '''Repack global measurement list into numpy arrays of desired format.
    '''

    N = len(all_Y)

    p = max( max(all_nsegs), 1 )
    nsegs = np.array(all_nsegs).reshape( 1,N )
    Y = np.array(all_Y).reshape( N, )
    sig_m  = np.array(all_sig_m).reshape( N,1 )
    L = np.array(all_L).reshape( 1,N )

    entry = np.zeros( (3*p, N) )
    for i,en in enumerate(all_entry):
        entry[:len(en[:]),i] = en[:]

    exit = np.zeros( (3*p, N) )
    for i,ex in enumerate(all_exit):
        exit[:len(ex[:]),i] = ex[:]

    nhat = np.array( all_nhat ).T
    kappa = np.array( all_kappa ).T
    return Y, sig_m, entry, exit, nhat, kappa, L, nsegs


def measure( strain, intersection_points, strain_dir ):
    '''Perform a single measurement integrating by quadrature over
    the strain function.
    '''

    nsegs = intersection_points.shape[0]//2

    kappa = list(strain_dir)
    entry,exit = [],[]
    Y = 0
    L = 0

    p1 = intersection_points[0,:] 
    p2 = intersection_points[1,:]
    
    length = np.linalg.norm(p2-p1)
    xray_direction = (p2-p1)/length
    nhat = list(xray_direction)
    theta = np.arccos( -xray_direction.dot(strain_dir) )
    sig_m = (np.pi - theta)*(1e-4/np.pi) + 1e-4 # higher scattering angles are more certain measurements

    for i in range( nsegs ):
        p1 = intersection_points[2*i,:] 
        p2 = intersection_points[2*i+1,:]
        entry.extend( list(p1) )
        exit.extend( list(p2) )
        length = np.linalg.norm(p2-p1)
        def func(s): 
            epsilon = strain(p1 + s*xray_direction)
            return strain_dir.dot( epsilon ).dot( strain_dir.T )
        Y += quad(func, 0, length)[0]
        L += length
    Y = Y/L
        
    return Y, sig_m, entry, exit, nhat, kappa, L, nsegs
