import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
from scipy.ndimage.morphology import binary_dilation

def voxels_to_polygon( image_stack, pixel_size, center=(0.5,0.5) ):
    '''Take a stack of images and produce a stack of shapely polygons.
    The images are interpreted as a solid shape with boundary along the pixel 
    exterior edge. Thus an image eith a single nonzero pixel will return a square 
    polygon with sidelength equal to the pixel_size.

    IN:
        image_stack: list of binary (1.0,0) numpy array 2d images each depicting
                     a single connected region of 1.0 surrounded by 0.0.
        pixel_size: The absolute pixel size of the input images. Used to make the
                    output polygons coordinates real spaced.
        center: the relative origin of the image, axis=0 is x and axis=1 is y
                increasing with increasingf index. For instance center=(0.5,0.5)
                will select the centre of the image as the orign.

    OUT:
        polygon_stack: list of shapely.geometry.polygons each representing the bound
                       of the corresponding input binary image.
    '''
    polygon_stack = [pixels_to_polygon(image, pixel_size, center) for image in image_stack]
    return polygon_stack

def pixels_to_polygon( image, pixel_size, center=(0.5,0.5) ):
    '''Take a single image and produce a shapely polygon.
    '''
    expanded_image = expand_image(image, factor=3)
    indices = get_image_boundary_index( expanded_image )
    coordinates = indices_to_coordinates(indices, pixel_size/3., center, expanded_image)
    polygon = shapely.geometry.Polygon(coordinates)
    #show_polygon_and_image(polygon, image, pixel_size, center) #<= DEBUG
    return polygon

def expand_image(image, factor):
    '''Expand 2d binary image so that each pixel is split by copying 
    into factor x factor number of pixels.
    '''
    expanded_image = np.repeat(image, factor, axis=1)
    expanded_image = np.repeat(expanded_image, factor, axis=0)
    return expanded_image

def get_image_boundary_index( image ):
    '''Find the pixel indices of the boundary pixels of a binary image.
    '''

    boundary_image = get_boundary_image( image )

    bound_indx = np.where( boundary_image==1 )
    ix,iy = bound_indx[0][0],bound_indx[1][0] # starting index
    indices = [(ix,iy)]
    while( not len(indices)==np.sum(boundary_image) ):
        # Walk around border and save boundary pixel indices
        mask = np.zeros(boundary_image.shape)
        mask[ np.max([0,ix-1]):ix+2, iy ] = 1
        mask[ ix, np.max([iy-1]):iy+2 ]   = 1
        neighbour_indx = np.where( boundary_image*mask )
        for ix,iy in zip( neighbour_indx[0], neighbour_indx[1]):
            if (ix,iy) not in indices:
                indices.append( (ix,iy) )
                break
    indices = sparse_indices( indices )
    return indices

def get_boundary_image( image ):
    '''Return a pixel image with 1 along the boundary if the assumed
    object in image.
    '''
    k = np.ones((3,3),dtype=int)
    dilation = binary_dilation( image==0, k, border_value=1 )
    boundary_image = dilation*image
    return boundary_image

def sparse_indices( indices ):
    '''Remove uneccesary nodes in the polygon (three nodes on a line is uneccesary).
    '''
    new_indices = []
    for i in range(0, len(indices)-1):
        if not (indices[i-1][0]==indices[i][0]==indices[i+1][0] or \
        indices[i-1][1]==indices[i][1]==indices[i+1][1]):
            new_indices.append(indices[i])
    return new_indices

def indices_to_coordinates(indices, pixel_size, center, image  ):
    '''Compute real space coordinates of image boundary form set of pixel indices. 
    '''
    dx = image.shape[1]*center[0]
    dy = image.shape[0]*center[1]
    coordinates = []
    for c in indices:
        # Verified by simulated nonsymmetric grain
        ycoord = pixel_size*(  c[1] + 0.5 - dx + (c[1]%3 - 1)*0.5 )
        xcoord = pixel_size*( -c[0] - 0.5 + dy - (c[0]%3 - 1)*0.5 )
        coordinates.append( (xcoord,ycoord) )
    return coordinates


def get_integral_paths( angles, ytrans, zpos, sample_polygon, show_geom=False ):
    '''Compute entry-exit points for a scanrange.
    '''

    # Instantiate lists to contain all measurements
    all_entry, all_exit, all_nhat, all_L, all_nsegs, bad_lines = [],[],[],[],[],[]
    xray_endpoints = get_xray_endpoints( sample_polygon )

    # Loop over all experimental settings
    for i,(ang,dty) in enumerate( zip( angles, ytrans ) ):

        # Translate and rotate the xray endpoints according to ytrans and angle
        c, s = np.cos( np.radians( -ang ) ), np.sin( np.radians( -ang ) )
        rotz = np.array([[c,-s],[s,c]])
        rx = rotz.dot( xray_endpoints + np.array([[0,0],[dty,dty]]) ) 
        xray_polygon = shapely.geometry.LineString( [ rx[:,0], rx[:,1] ] )

        # compute the intersections between beam and sample in sample coordinates
        intersection_points = get_intersection( xray_polygon, sample_polygon, zpos  )
    
        if intersection_points is None:
            # If a measurement missed the sample or graced a corner, we skipp ahead
            bad_lines.append(i)
            continue
        else:
            # make a measurement at the current setting
            entry, exit, nhat, L, nsegs = get_quanteties( intersection_points )

            # save the measurement results in global lists
            all_entry.append( entry )
            all_exit.append( exit )
            all_nhat.append( nhat )
            all_L.append( L )
            all_nsegs.append( nsegs )

            if show_geom:
                # Show sample and xray polygons
                print('entry ',entry)
                print('exit ',exit)
                print('nhat ',nhat)
                print('L ',L)
                xc,yc = xray_polygon.xy
                plt.figure(figsize=(11,8))
                plt.scatter(entry[0::3], entry[1::3],c='k',zorder=200,label='entry')
                plt.scatter(exit[0::3], exit[1::3],c='b',zorder=200,label='exit')
                plt.plot(xc,yc,c='y',label='Beam')
                plt.arrow(0, 0, 20*nhat[0],20*nhat[1], head_width=2, color='r', zorder=100,label=r'$\hat{n}$')
                xc,yc = sample_polygon.exterior.xy
                plt.fill(xc,yc,color='gray',label='Grain',zorder=1)
                plt.title('L='+str(L)+' angle='+str(ang)+r'$^o$,   dty='+str(dty)+',   z_translation='+str(zpos)+ 'nsegs='+str(nsegs))
                plt.axis('equal')
                plt.xlabel('x')
                plt.ylabel('y')
                
                xcircle = np.linspace(-dty, dty, 100)
                ycircle = np.sqrt( dty**2 - xcircle**2 )
                plt.plot( xcircle, ycircle, c='g' )
                plt.plot( xcircle, -ycircle, c='g', label='circle with R=dty' )
                plt.grid(True)
                plt.legend()
                plt.show()

    # repack lists of measurements into numpy arrays of desired format
    entry, exit, nhat, L, nsegs = repack(all_entry, all_exit, all_nhat, all_L, all_nsegs)

    return  entry, exit, nhat, L, nsegs, bad_lines


def get_xray_endpoints( sample_polygon ):
    '''Calculate endpoitns of xray line segement. The lenght of the 
    line segment is adapted to make sure xray always convers the full
    length of the sample.
    '''
    xc, yc = sample_polygon.exterior.xy
    xmin = np.min( xc )
    xmax = np.max( xc )
    ymin = np.min( yc )
    ymax = np.max( yc )
    D = np.sqrt( (xmax-xmin)**2 + (ymax-ymin)**2 )
    return np.array([ [-1.1*D, 1.1*D], [0,0] ])

def get_intersection( xray_polygon, sample_polygon, z ):
    '''Compute the 3d coordinates of intersection between xray and
    sample.
    '''
    intersection = sample_polygon.intersection( xray_polygon )
    if intersection.is_empty or isinstance(intersection, shapely.geometry.point.Point):
        # we missed the sample with the beam
        intersection_points = None
    elif isinstance(intersection, shapely.geometry.linestring.LineString):
        # we got a single line segment intersection
        intersection_points = np.zeros( (2,3) )
        intersection_points[:2,:2] = np.array( intersection.xy ).T
        intersection_points[:,2] = z
    elif isinstance(intersection, shapely.geometry.multilinestring.MultiLineString):
        # we got multiple line segments intersection
        intersection_points = np.zeros( (2*len(intersection.geoms),3) )
        for i,line_segment in enumerate(intersection.geoms):
            intersection_points[2*i:2*(i+1),:2] = np.array( line_segment.xy ).T
        intersection_points[:,2] = z 
    return intersection_points

def get_quanteties( intersection_points ):
    nsegs = intersection_points.shape[0]//2
    entry,exit = [],[]
    p1 = intersection_points[0,:] 
    p2 = intersection_points[1,:]
    nhat = list( (p2-p1)/np.linalg.norm(p2-p1) )

    L = 0
    for i in range( nsegs ):
        p1 = intersection_points[2*i,:] 
        p2 = intersection_points[2*i+1,:]
        entry.extend( list(p1) )
        exit.extend( list(p2) )
        length = np.linalg.norm(p2-p1)
        L += length

    return entry, exit, nhat, L, nsegs

def repack( all_entry, all_exit, all_nhat, all_L, all_nsegs ):
    '''Repack global measurement list into numpy arrays of desired format.
    '''
    N = len( all_L )

    p = max( max(all_nsegs), 1 )
    nsegs = np.array(all_nsegs).reshape( 1,N )
    L = np.array(all_L).reshape( 1,N )

    entry = np.zeros( (3*p, N) )
    for i,en in enumerate(all_entry):
        entry[:len(en[:]),i] = en[:]

    exit = np.zeros( (3*p, N) )
    for i,ex in enumerate(all_exit):
        exit[:len(ex[:]),i] = ex[:]

    nhat = np.array( all_nhat ).T
    return entry, exit, nhat, L, nsegs

def show_polygon_and_image( polygon, image, pixel_size, center ):
    '''Plot a image and polygon for debugging purposes
    '''
    fig,ax = plt.subplots(1, 2, figsize=(12,6))
    fig.suptitle('Center at '+str(center))
    xc,yc = polygon.exterior.xy
    xcenter = image.shape[1]*pixel_size*center[0]
    ycenter = image.shape[0]*pixel_size*center[1]
    ax[0].imshow(image,cmap='gray')
    ax[0].set_title('Pixel image')
    ax[0].arrow( int(image.shape[1]*center[0]), int(image.shape[0]*center[1]), \
        image.shape[0]//4, 0, color='r', head_width=0.15 ) # y
    ax[0].text( int(image.shape[1]*center[0])+image.shape[1]//4, int(image.shape[0]*center[1])+0.25, \
        'y',color='r' )
    ax[0].arrow( int(image.shape[1]*center[0]), int(image.shape[0]*center[1]), \
        0, -image.shape[1]//4, color='r', head_width=0.15 ) # x
    ax[0].text( int(image.shape[1]*center[0])+0.25, int(image.shape[0]*center[1])-image.shape[1]//4,  \
        'x',color='r' )
    ax[1].set_title('Polygon representation')
    ax[1].fill(xc,yc,c='gray',zorder=1)
    ax[1].scatter(xc,yc,c='r',zorder=2)
    ax[1].grid(True)
    ax[1].scatter(0, 0 , c='b',zorder=3)
    ax[1].set_xlim([ -xcenter, image.shape[1]*pixel_size - xcenter ])
    ax[1].set_ylim([ -ycenter, image.shape[0]*pixel_size - ycenter ])
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    plt.show()

