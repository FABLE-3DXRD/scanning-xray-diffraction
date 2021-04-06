import numpy as np
import copy
import matplotlib.pyplot as plt

# def match_and_label( pixelated_grains, id11_grains ):
#     '''Match and label all grains in a 3d volume so that all the topology
#     of any single grain can be easily extracted. This method returns produces
#     resorted lists of lists (pixelated_grains, id11_grains) such that the same
#     index corresponds to the same grain throughout slices.
#     in:
#         pixelated_grains: list of lists containing 2d numpy arrays of each grain
#                           topology as a binary mask. Outer list is for z-slice inner
#                           goes over all grains in the slice.

#         ubis:             list of lists with imageD11 grain objects. Outer list is for 
#                           z-slice inner goes over all grains in the slice.
#     out:
#         matched_pixelated_grains: same as pixelated_grains but sorted such that each index
#                                   in the inner grain list matches the same grain in the next
#                                   slice.
#         matched_id11_grains:      same as ubis but sorted such that each index in the inner 
#                                   grain list matches the same grain in the next slice.
#     '''

#     # We find only as many grains as we have in the reference slice
#     # which is here selected as the central slice. To have multiple
#     # layers of grains in z, we need to loop and reset reference slices.
#     # This is a TODO if we have such data for this project.
#     number_zslices = len(pixelated_grains)
#     reference_slice = number_zslices//2
#     number_reference_grains = len(pixelated_grains[reference_slice])

#     # Locate the grains existing in reference_slice and rebuild pixelated_grains and id11_grains
#     matched_pixelated_grains = [ [None]*number_reference_grains for _ in range(number_zslices)]
#     matched_id11_grains = [ [None]*number_reference_grains for _ in range(number_zslices)]

#     for gi in range( number_reference_grains ):
        
#         # The slice matcher object helps to do the matching
#         slm = SliceMatcher( gi )
#         slm.set_reference(  pixelated_grains[reference_slice][gi], id11_grains[reference_slice][gi] )
        
#         # Track grain upwards in positive z
#         for zi in range( reference_slice+1, number_zslices, 1 ):
#             indx = slm.match( id11_grains[zi], pixelated_grains[zi] )
#             if indx is None: 
#                 break
#             else:
#                 matched_pixelated_grains[zi][gi] = copy.deepcopy( pixelated_grains[zi][indx] )
#                 matched_id11_grains[zi][gi] = copy.deepcopy( id11_grains[zi][indx] )

#         # Reset reference to central slice before looping in negative z 
#         slm.reset_reference()
#         matched_pixelated_grains[reference_slice][gi] = copy.deepcopy( pixelated_grains[reference_slice][gi] )
#         matched_id11_grains[reference_slice][gi] = copy.deepcopy( id11_grains[reference_slice][gi] )

#         # Track grain downwards in negative z
#         for zi in range( reference_slice-1, -1, -1 ):
#             indx = slm.match( id11_grains[zi], pixelated_grains[zi] )
#             if indx is None: 
#                 break
#             else:
#                 matched_pixelated_grains[zi][gi] = copy.deepcopy( pixelated_grains[zi][indx] )
#                 matched_id11_grains[zi][gi] = copy.deepcopy( id11_grains[zi][indx] )

#     return matched_pixelated_grains, matched_id11_grains


# class SliceMatcher(object):
    
#     def __init__( self, index ):
#         self.index = index 
#         self.ref_grain_shape = None
#         self.ref_grain = None

#     def set_reference(self, ref_grain_shape, ref_grain):
#         if self.ref_grain==None:
#             self.original_ref_grain = copy.deepcopy(ref_grain)
#             self.original_ref_grain_shape = copy.deepcopy(ref_grain_shape)
#         self.ref_grain_shape = ref_grain_shape
#         self.ref_grain = ref_grain

#     def reset_reference(self):
#         self.ref_grain_shape = copy.deepcopy(self.original_ref_grain_shape)
#         self.ref_grain = copy.deepcopy(self.original_ref_grain)

#     def match_from_overlap( self, grain_shapes ):
#         overlap = [np.sum( self.ref_grain_shape*gs ) for gs in grain_shapes]
#         candidate_indxs = np.where( np.array(overlap)>0 )[0]
#         return candidate_indxs

#     def match_from_u( self, grains, candidate_indxs ):
#         angdiffs = [ np.sum( np.degrees( self.ref_grain.u.T.dot(grains[i].u) ) ) for i in candidate_indxs]
#         indx     = np.argmin( angdiffs )
#         if angdiffs[indx] > 10: # total of 10 degree rotation defines new grain.
#             return None
#         else:
#             return candidate_indxs[ indx ]

#     def match( self, grains, grain_shapes ):
#         candidate_indxs = self.match_from_overlap( grain_shapes )
#         if len(candidate_indxs)>0:
#             return self.match_from_u( grains, candidate_indxs )
#         else:
#             return None


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




