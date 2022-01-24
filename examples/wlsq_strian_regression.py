
import numpy as np
import pickle
import matplotlib.pyplot as plt

"""After running diffraction_analysis.py it is time to do some regression

In this script we do weighted least squares strain regression.
"""

if __name__=="__main__":
    # The meta parameters are the same still as in diffraction_analysis.py
    ystep     =  5.                           # Translation stepsize in microns.
    zpos      = np.array(range(1,13))*ystep   # coordinates of scanned positions in z

    # We load the vector data we previously computed in diffraction_analysis.py
    from s3dxrd.utils.save import load_vectors
    vectors = load_vectors("./example_results/vectors.pkl")

    # To do the regression we need to define a mesh of elements wehre the strain field lives, for this the mesher module is nice
    from s3dxrd.utils.mesher import mesh_from_polygon, mesh_to_pixels

    # Now it is time to perform the regression for strain, we will need the s3dxrd.regression.wlsq module.
    from s3dxrd.regression.wlsq import trust_constr_solve

    # Now we are ready to reconstruct, let's do the middle slice of the first grain for instace,
    # It may take some time (15 seconds or so) to build the projection matrix, use verbose=True so see the
    # progress as the system of equations gets assembled.

    # we mesh the polygonal slice the first number is a grain identifyer and the second a slice
    # index in z. We take grain number 1 slice number 7 here.
    mesh = mesh_from_polygon(ystep, vectors["polygons"]["1"]["7"])

    # and mask out the relevant meaurements by grain index and z-position.
    mask = (vectors["measurement_grain_map"]==1)*(vectors["entry"][2,:]==zpos[7])

    # Next we call the regression code passing along only the masked out measurements.
    # if you want you could run on several cores changing the nprocs=1 argument
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    wlsq_strain = trust_constr_solve( mesh, vectors["kappa"][:,mask], vectors["Y"][mask],
                                    vectors["entry"][:,mask], vectors["exit"][:,mask],
                                    vectors["nhat"][:,mask], 1./vectors["sig_m"][mask].flatten(),
                                    ystep, grad_constraint=5*(1e-4),
                                    maxiter=5, verbose=True, nprocs=1)
    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)

    # We can plot the resulting field by converting it into pixelated images using mesh_to_pixels()
    # The save module can help with saving to paraview
    from s3dxrd.utils.save import as_vtk_voxel_volume
    fig,ax = plt.subplots(2,3)
    titles = ["XX","YY","ZZ","YZ","XZ","XY"]
    voxel_strains = []
    for s,a,t in zip(wlsq_strain,ax.flatten(),titles):
        voxel_strains.append( mesh_to_pixels(mesh, ystep, (25,25), s) )
        a.imshow(voxel_strains[-1])
        a.set_title("Strain "+t+" at z="+str(zpos[7])+"microns")
    plt.show()

    voxel_strains_3d = [np.expand_dims(vs,axis=1) for vs in voxel_strains]
    as_vtk_voxel_volume("./example_results/strain_slice", voxel_strains_3d, titles)



