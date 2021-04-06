import numpy as np
import copy
import matplotlib.pyplot as plt

def match_and_label( pixelated_grains, id11_grains, angthres=3.0 ):
    """Segment a stack of 2d grain slices into 3d grains by orientation and z-overlap.

    This function iterates over stacks for pixelated grain masks and corresponding Id11
    grain objects to create a numpy 3d volume where each voxel has a uniqe integer number
    corresponding to the grain index. Void regions are set to 0. The index at a voxel
    corresponds to a Id11 grain object in an output dictionary of Id11 grains.

    Grains are considered to be one and the same if and only if:
        #1: They have at least one overlaping voxel in z.
        #2: They have orientation matrices resutling in a smaller angular deviation (dgrs)
            than angthres. This number represents the sum of angular devation at each
            orientation vector formed by taking the arccos of the three scalar products.
            
    Args:
        pixelated_grains (:obj:`list` of :obj:`lists` of :obj:`numpy arrays`): Pixelated masks of the grains
        id11_grains (:obj:`list` of :obj:`lists` of :obj:`ImageD11 Grain`): ImageD11 grain representations
            corresponding to the masks in pixelated_grains.
        angthres (:obj:`float`): Criteria threshold for mapping grains across z. Defaults to 3.0.
    
    Returns:
        sample (:obj:`numpy array`): Voxelated representation of full sample. the value at sample[i,j,k]
        is ```0``` if void and otherwise equal to the index of the grain in ```grains``` 
        grains (:obj:`dict` of :obj:`dict` of :obj:`ImageD11 Grain`): ImageD11 grain representations
            corresponding to ```sample``` index values such that grains["3"]["7"] gives the 7th slice if grain
            with index 3 in ```sample```.

    """

    grain_indx = 1
    grains = {}
    dx,dy = pixelated_grains[0][0].shape
    sample = np.zeros( (dx, dy, len(pixelated_grains) ), dtype=int )

    for j in range(len(pixelated_grains[0])):
        mask = pixelated_grains[0][j]
        sample[mask,0] = grain_indx
        grains[str(grain_indx)] = { str(0) : id11_grains[0][j] }
        grain_indx += 1

    for i in range(1, sample.shape[2]):
        for j in range(len(pixelated_grains[i])):
            found_match = False
            mask = pixelated_grains[i][j]
            overlap = ( sample[:,:,i-1] * mask ) > 0
            if np.sum( overlap ) > 0:
                indices = np.unique(sample[overlap,i-1])
                for indx in indices[indices!=0]:
                    u1 = grains[str(indx)][str(i-1)].u
                    u2 = id11_grains[i][j].u
                    angdiff = np.sum( [ np.degrees( np.arccos(u1[:,k].dot(u2[:,k])) ) for k in range(3) ] )
                    if angdiff < angthres:
                        sample[mask,i] = indx
                        found_match = True
                        grains[str(indx)][str(i)] = id11_grains[i][j]
                        break

            if not found_match:
                sample[mask,i] = grain_indx
                grains[str(grain_indx)] = { str(i) : id11_grains[i][j] }
                grain_indx += 1
            
    return sample, grains




