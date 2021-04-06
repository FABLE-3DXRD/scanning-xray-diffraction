import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as shapelyPolygon
from shapely.geometry import Point as shapelyPoint

def mesh_from_polygon( w, poly ):
    """Generate a quadratile 2d mesh over a Shapely polygon object.

    Args:
        w (:obj:`float`): Element side length.
        poly (:obj:`Shapely Polygon`): Polygon object to be meshed. The output mesh 
            follows the :obj:`poly` coordinate system.
    
    Returns:
        (:obj:`numpy array`) The nodal coordinates of each element in the mesh.

    """
    x = poly.exterior.xy[0]
    y = poly.exterior.xy[1]
    xmin,xmax = np.min(x), np.max(x)
    ymin,ymax = np.min(y), np.max(y)

    ll = [xmin, ymin]
    mesh = []
    while(ll[1]<=ymax-w):
        if poly.contains( shapelyPoint([(ll[0]+w/2., ll[1]+w/2.)]) ):
            mesh.append([ll[0], ll[1], ll[0]+w, ll[1], ll[0]+w, ll[1]+w, ll[0], ll[1]+w])
        if ll[0] < xmax - w:
            ll[0] = ll[0] + w
        else:
            ll[0] = xmin
            ll[1] = ll[1] + w
    return np.array(mesh)

def mesh_to_pixels( mesh, element_width, shape, values ):
    """Generate pixelated map from a mesh coordinate matrix.

    Args:
        mesh (:obj:`numpy array`): Element nodal coordinates
        element_width (:obj:`float`): Element side length.
        shape  (:obj:`tuple`): Output array shape
        values (:obj:`numpy array`): Per elemetn values to fill the output array with.

    Returns:
        (:obj:`numpy array`) Pixelated map of input field stored in values.

    """
    mask = np.full(shape, np.nan)
    for i,(xc,yc) in enumerate(zip( np.mean(mesh[:,0::2],axis=1), np.mean(mesh[:,1::2],axis=1) )):
        xindx =  -np.ceil( -xc/element_width ).astype(int) + shape[0]//2
        yindx =   np.ceil( yc/element_width ).astype(int) + shape[1]//2
        mask[yindx,xindx] = values[i]
    return mask

def show_mesh( mesh, poly ):
    print(mesh.shape)
    x = poly.exterior.xy[0]
    y = poly.exterior.xy[1]
    for e in mesh:
        for i in [0,2,4]:
            plt.plot([e[i],e[i+2]],[e[i+1],e[i+3]],'-ko')
        plt.plot([e[-2],e[0]],[e[-1],e[1]],'-ko')
    for i in range(len(x)-1):
        plt.plot([x[i],x[i+1]],[y[i],y[i+1]],'k')
    plt.plot([x[-1],x[0]],[y[-1],y[0]],'k')
    plt.grid(True)
    plt.show()