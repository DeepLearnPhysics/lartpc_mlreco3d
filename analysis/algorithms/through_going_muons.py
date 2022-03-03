from collections import OrderedDict
from turtle import update
from sklearn.decomposition import PCA

from analysis.classes.ui import FullChainEvaluator, FullChainPredictor
from analysis.decorator import evaluate
from analysis.classes.particle import match

from pprint import pprint
import time
import numpy as np

@evaluate(['acpt_muons_cells', 'acpt_muons'], mode='per_batch')
def through_going_muons(data_blob, res, data_idx, analysis_cfg, cfg):
    """
    Selection of anode-cathode crossing muons
    ==========================================

    The idea is to select muons whose x-projected length matches
    with the detector size in the x-direction. Then we claim to
    have found an anode-cathode crossing muon. We split it into
    segments using a cubic 3D binning of the detector volume.
    For each segment we perform PCA to estimate the direction,
    hence dx, and we sum reco charge to get dQ.

    This heuristic relies only on semantic segmentation and
    particle clustering predictions.

    Note: we might need to use PID to weed out pion tracks.

    Output
    ======
    Two CSV files should contain respectively:
    - array of measurements of dQ/dx for each non-empty cell (`acpt_muons_cells.csv`)
    - array of detected muons and related information (`acpt_muons.csv`)
    """
    # Set default fieldnames and values. (Needed for logger to work)
    muon_cells, muons = [], []
    deghosting = analysis_cfg['analysis']['deghosting']
    primaries = analysis_cfg['analysis']['match_primaries']
    spatial_size = analysis_cfg['analysis']['processor_cfg']['spatial_size']
    selection_threshold = analysis_cfg['analysis']['processor_cfg']['selection_threshold']
    bin_size = analysis_cfg['analysis']['processor_cfg']['bin_size']
    data = analysis_cfg['analysis']['processor_cfg'].get('data', False)

    if not data:
        predictor = FullChainEvaluator(data_blob, res, cfg, analysis_cfg, deghosting=deghosting)
    else:
        predictor = FullChainPredictor(data_blob, res, cfg, analysis_cfg, deghosting=deghosting)

    image_idxs = data_blob['index']

    # TODO check that 0 is actually drift direction
    # TODO check that 2 is actually vertical direction
    x, y, z = 0, 1, 2

    pca = PCA(n_components=2)

    # 1. Select particles with Track semantics
    # 2. Check that both endpoints are within some_threshold of boundaries
    # 3. Exclude particles attached to a Michel particle
    for i, index in enumerate(image_idxs):
        pred_particles = predictor.get_particles(i, primaries=False)
        if not data:
            true_particles = predictor.get_true_particles(i, primaries=False)

            # Match true particles to predicted particles
            true_ids = np.array([p.id for p in true_particles])
            matched_particles, _, _ = match(true_particles, pred_particles,
                                            min_overlap=0.1)

        for p in pred_particles:
            if p.semantic_type != 1: continue
            x_coords = p.points[:, x]
            projected_x_length = x_coords.max() - x_coords.min()
            if projected_x_length > spatial_size - selection_threshold:
                # We are keeping this candidate, see if it is a match
                # for the records keeping

                # T0 correction, assuming T0 is minimal drift coordinate
                coords = p.points
                coords[:, x] = coords[:, x] - x_coords.min()
                update_dict = {
                    'index': index,
                    'pred_particle_type': p.pid,
                    'pred_particle_is_primary': p.is_primary,
                    'pred_particle_size': p.size,
                    'projected_x_length': projected_x_length,
                    'theta_yz': np.arctan2((coords[:, y].max() - coords[:, y].min()),(coords[:, z].max()-coords[:, z].min())),
                    'theta_xz': np.arctan2((coords[:, x].max() - coords[:, x].min()),(coords[:, z].max()-coords[:, z].min())),
                    'matched': len(p.match),
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

                # Bin track in segments
                bins = np.arange(0, spatial_size, bin_size)
                y_inds = np.digitize(coords[:, y], bins)
                z_inds = np.digitize(coords[:, z], bins)
                x_inds = np.digitize(coords[:, x], bins)

                # Assuming x binning is a multiple
                for y_idx in np.unique(y_inds):
                    for z_idx in np.unique(z_inds):
                        for x_idx in np.unique(x_inds):
                            cell = (y_inds == y_idx) & (z_inds == z_idx) & (x_inds == x_idx)
                            if np.count_nonzero(cell) < 2: continue
                            coords_pca = pca.fit_transform(coords[cell])[:, 0]
                            update_dict = OrderedDict({
                                'index': index,
                                'pred_particle_type': p.pid,
                                'pred_particle_is_primary': p.is_primary,
                                'pred_particle_size': p.size,
                                'cell_dQ': p.depositions[cell].sum(),
                                'cell_dN': np.count_nonzero(cell),
                                'cell_dx': coords_pca.max() - coords_pca.min(),
                                'cell_ybin': y_idx,
                                'cell_zbin': z_idx,
                                'cell_xbin': x_idx
                            })
                            muon_cells.append(update_dict)

    return [muon_cells, muons]
