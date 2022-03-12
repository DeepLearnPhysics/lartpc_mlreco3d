from collections import OrderedDict
from turtle import update
from sklearn.decomposition import PCA

from analysis.classes.ui import FullChainEvaluator, FullChainPredictor
from analysis.decorator import evaluate
from analysis.classes.particle import match

from pprint import pprint
import time
import numpy as np
from scipy.spatial.distance import cdist


@evaluate(['stopping_muons_cells', 'stopping_muons'], mode='per_batch')
def stopping_muons(data_blob, res, data_idx, analysis_cfg, cfg):
    """
    Selection of stopping muons
    ===========================

    To convert dQ/dx from ADC/cm to MeV/cm. We want a sample as pure
    as possible, hence the option to enforce the presence of a Michel
    electron at the end of the muon.

    Configuration
    =============

    """
    muon_cells, muons = [], []

    deghosting          = analysis_cfg['analysis']['deghosting']
    processor_cfg       = analysis_cfg['analysis'].get('processor_cfg', {})

    spatial_size        = processor_cfg['spatial_size']
    #selection_threshold = processor_cfg['selection_threshold']
    bin_size            = processor_cfg['bin_size']
    # Whether we are running on MC or data
    data                = processor_cfg.get('data', False)
    # Whether to restrict to tracks that are close to Michel voxels
    # threshold =-1 to disable, otherwise it is the threshold below which we consider the track
    # might be attached to a Michel electron.
    Michel_threshold    = processor_cfg.get('Michel_threshold', -1)
    # Whether to enforce PID constraint (predicted as muon only)
    pid_constraint      = processor_cfg.get('pid_constraint', False)
    # Avoid hardcoding labels
    muon_label       = processor_cfg.get('muon_label', 2)
    track_label      = processor_cfg.get('track_label', 1)

    # Initialize analysis differently depending on data/MC setting
    if not data:
        predictor = FullChainEvaluator(data_blob, res, cfg, analysis_cfg, deghosting=deghosting)
    else:
        predictor = FullChainPredictor(data_blob, res, cfg, analysis_cfg, deghosting=deghosting)

    image_idxs = data_blob['index']

    # TODO check that 0 is actually drift direction
    # TODO check that 2 is actually vertical direction
    x, y, z = 0, 1, 2

    pca = PCA(n_components=2)

    for i, index in enumerate(image_idxs):
        pred_particles = predictor.get_particles(i, primaries=False)

        # Match with true particles if available
        if not data:
            true_particles = predictor.get_true_particles(i, primaries=False)
            # Match true particles to predicted particles
            true_ids = np.array([p.id for p in true_particles])
            matched_particles, _, _ = match(true_particles, pred_particles,
                                            min_overlap=0.1)

        # Loop over predicted particles
        for p in pred_particles:
            if p.semantic_type != track_label: continue
            coords = p.points

            # Check for presence of Michel electron
            attached_to_Michel = False
            closest_point = None
            for p2 in pred_particles:
                if p2.semantic_type != 2: continue
                d =  cdist(p.points, p2.points)
                if d.min() >= Michel_threshold: continue
                attached_to_Michel = True
                closest_point = d.min(axis=1).argmin()

            if not attached_to_Michel: continue

            # If asked to check predicted PID, exclude non-predicted-muons
            if pid_constraint and p.pid != muon_label: continue

            # PCA to get a rough direction
            coords_pca = pca.fit_transform(p.points)[:, 0]
            # Make sure where the end vs start is
            # if end == 0 we have the right bin ordering, otherwise might need to flip when we record the residual range
            end = np.argmin([((coords[coords_pca.argmin(), :] - closest_point)**2).sum(), ((coords[coords_pca.argmax(), :] - closest_point)**2).sum()])

            # Record the stopping muon
            update_dict = {
                'index': index,
                'pred_particle_type': p.pid,
                'pred_particle_is_primary': p.is_primary,
                'pred_particle_size': p.size,
                #'projected_x_length': projected_x_length,
                'theta_yz': np.arctan2((coords[:, y].max() - coords[:, y].min()),(coords[:, z].max()-coords[:, z].min())),
                'theta_xz': np.arctan2((coords[:, x].max() - coords[:, x].min()),(coords[:, z].max()-coords[:, z].min())),
                'matched': len(p.match),
                'pca_length': coords_pca.max() - coords_pca.min(),
                #'t0': t0,
                'true_pdg': -1,
                'true_size': -1,
                }
            if not data and len(p.match)>0:
                m = np.array(true_particles)[true_ids == p.match[0]][0]
                update_dict.update({
                    'true_pdg': m.pid,
                    'true_size': m.size
                })
            muons.append(OrderedDict(update_dict))
            track_dict= update_dict

            # Split into segments and compute local dQ/dx
            bins = np.arange(coords_pca.min(), coords_pca.max(), bin_size)
            bin_inds = np.digitize(coords_pca, bins)

            # spatial_bins = np.arange(0, spatial_size, spatial_bin_size)
            # y_inds = np.digitize(coords[:, y], bins)
            # z_inds = np.digitize(coords[:, z], bins)
            # x_inds = np.digitize(coords[:, x], bins)
            for i in np.unique(bin_inds):
                mask = bin_inds == i
                if np.count_nonzero(mask) < 2: continue
                # Repeat PCA locally for better measurement of dx
                pca_axis = pca.fit_transform(p.points[mask])
                dx = pca_axis[:, 0].max() - pca_axis[:, 0].min()
                update_dict = OrderedDict({
                    'index': index,
                    'cell_dQ': p.depositions[mask].sum(),
                    'cell_dN':  np.count_nonzero(mask),
                    'cell_dx': dx,
                    'cell_bin': i,
                    'cell_residual_range': (i if end == 0 else len(bins)-i-1) * bin_size,
                    'nbins': len(bins)
                })
                update_dict.update(track_dict)
                muon_cells.append(update_dict)

    return [muon_cells, muons]
