import numpy as np
import matplotlib.pyplot as plt
from ImageD11 import columnfile, parameters, grain
from s3dxrd.utils import peak_mapper, reconstruct_grainshapes
from xfab import tools

class RawMeasurements(object):
    """Container of semi-raw diffraction data and grain properties.

    Performs diffraction peak mapping and grain topology reconstruction
    working of ImageD11 file formats. These files are already proccessed in
    the sense that the grains have been indexed and detector tilt has
    been calibrated for etc. In scanning-3DXRD each layer (x-y) is scanned
    independently of z translations. Thus the input is a stacks of
    measurements, one file per sample z-translation. The coordinate system
    and experimental setup is described in:

        Henningsson, N. A., Hall, S. A., Wright, J. P. & Hektor, J. 
        (2020). J. Appl. Cryst. 53, 314-325.
        https://doi.org/10.1107/S1600576720001016

    Attributes:
        zpos (list of float): Sample z-translations corresponding to grain_slices
            indices and peak_stack indices.
        grain_slices (list of lists of :obj:`grain`): Loaded ImageD11 grain '.map'
            or '.ubi' files. Each z-translation has a list of grains present in
            the slice. The map files are loaded via the ImageD11 grain module.
        peak_stack (list of :obj:`columnfile`): Loaded ImageD11 peak files. Each
            instance of the peak stack list contains a columnfile describing
            diffraction peaks originating from correpsonding z-translation. The 
            peak files are loaded via the ImageD11 columnfile module.
        params (:obj:`parameters`): Loaded ImageD11 parameter file. This object
            gives access to meta parameters of the experiment, such as detector tilt,
            sample to detector distance, and crystal cell parameters. The parameter
            file is loaded via the ImageD11 parameters module.
        omegastep (float): Rotation stage readout intervall in degrees. Each recorded
            frame in the experiment is was integrated over this intervall.
        ystep (float): y-translation stepsize between measurements. Normally the same
            as the beamsize.
        number_y_scans (float): The number of scanned y-positions made in each z-layer.
        ymin (float): Minium y-coordinate seen by the detector. These values differ from
            what is in the orginal columnfile in the case when the columnfile do not have
            a symmetric scanning intervall over y=0.
        ymax (float): Maximum y-coordinate seen by the detector. These values differ from
            what is in the orginal columnfile in the case when the columnfile do not have
            a symmetric scanning intervall over y=0.
        grain_topology_mask (list of 2D numpy arrays): Grain topology in binarized form.
            Each instacne of the list contains the grain shape at the corresponding
            z-slice. This attribute starts as None, and is set as an result of an active
            action.
    """

    def __init__(self, 
                flt_paths,
                zpos,
                param_path,
                ubi_paths,
                omegastep, 
                ymin, 
                ymax, 
                ystep):
        """Container of semi-raw diffraction data and grain properties.

        Args:
            zpos (list of float): Sample z-translations corresponding to flt_paths.
            ubi_paths (str): Absolute paths to ID11 grain files.
            flt_paths (list of str): Absolute paths to ID11 column files.
            param_path (str): Absolute path to ID11 parameter file.
            omegastep (float): Rotation stage readout intervall in degrees.
            ystep (float): stepsize of sample y-translation between measurements
            ymin (float): Minium y-coordinate seen by the detector, 
                as given in ID11 column file.
            ymax (float): Maximum y-coordinate seen by the detector, 
                as given in ID11 column file.
        """
        self.zpos = zpos
        self.grain_slices = [ grain.read_grain_file( ubip ) for ubip in ubi_paths ]

        for grs in self.grain_slices:
            for gr in grs:
                gr.u = tools.ubi_to_u( gr.ubi )

        self.peak_stack = [columnfile.columnfile(flt) for flt in flt_paths]
        self.tot_nbr_peaks = sum( [peaks.nrows for peaks in self.peak_stack] )
        self.params = parameters.read_par_file( param_path )
        self.omegastep = omegastep
        self.ystep = ystep
        self.number_y_scans = np.round((abs(ymax-ymin)/ystep)).astype(int)+1
        for p in self.peak_stack: 
            p.dty[:] = p.dty - ymin - self.ystep*(self.number_y_scans//2)
        self.ymin = -ystep*(self.number_y_scans//2)
        self.ymax =  ystep*(self.number_y_scans//2)
        self.grain_topology_mask = None

        self.params.parameters['ystep'] = self.ystep
        self.params.parameters['ymin'] = self.ymin
        self.params.parameters['t_x'] = None
        self.params.parameters['t_y'] = None
        self.params.parameters['t_z'] = None

    def map_peaks(self, hkltol, nmedian ):
        """Perform Grain centroid refinement and diffraction peak to grain mappings.

        This method refines the semi-raw diffraction data contained by the input 
        peak files. The grains of self.grain_slices will be given masks and refined
        average orientation and strain. The columnfiles of the peak_stack will be
        recomputed based on the centroid of the claimer grain.

        Note:
            This function mutates the attributes of the class and has no return.

        Args:
            hkltol (float): Tolerance on Miller indices (hkl) to match a peak.
            nmedian (float): Outlier threshold. If any of the diffraction angles 
                (2*theta, omega, eta) are more than nmedian times deviating from the 
                median deviation (between model and measurement) the peak is considered
                an outlier and removed.
        """
        for gs, flt, dtz in zip( self.grain_slices, self.peak_stack, self.zpos ):
            peak_mapper.map_peaks(flt, gs, self.params, self.omegastep, \
                                        hkltol, nmedian, self.ymin, self.ystep, self.number_y_scans)

    def reconstruct_grain_topology(self, rcut):
        """Perform Filtered Back Projection to approximate grain shapes.

        The recorded intensity of the diffraction peaks are used to create
        a grain topology map. Each grain is reconstructed on a pixelated grid,
        and thresholded to a binary 1 - 0, grain or no grain, map.

        Note:
            This function mutates the attributes of the class and has no return.

        Args:
            rcut (float): Threshold for segementing out grain. Reconstructed 
            relative intensity of less than rcut is considered void, and intensity
            higher than rcut is considered grain.
        """
        self.grain_topology_mask=[]
        for gs, flt, dtz in zip( self.grain_slices, self.peak_stack, self.zpos ):
            grain_topology_mask = reconstruct_grainshapes.FBP_slice(gs, flt,       \
                                        self.omegastep, rcut, self.ymin, \
                                        self.ystep, self.number_y_scans)                   
            self.grain_topology_mask.append( grain_topology_mask )
