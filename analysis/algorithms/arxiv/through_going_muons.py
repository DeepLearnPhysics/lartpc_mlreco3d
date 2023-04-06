from collections import OrderedDict
from turtle import update
from sklearn.decomposition import PCA

from analysis.classes.predictor import FullChainPredictor
from analysis.classes.evaluator import FullChainEvaluator
from lartpc_mlreco3d.analysis.algorithms.arxiv.decorator import evaluate
from analysis.algorithms.selections.flash_matching import find_true_time, find_true_x

from pprint import pprint
import time
import numpy as np
from scipy.spatial.distance import cdist


def must_invert(x, invert_regions):
    for r in invert_regions:
        if x >= r[0] and x <= r[1]:
            return True
    return False



@evaluate(['acpt_muons_cells', 'acpt_muons'])
def through_going_muons(data_blob, res, data_idx, analysis_cfg, cfg):
    """
    Selection of through going muons
    =================================

    1. Anode-cathode-piercing tracks:

    The idea is to select muons whose x-projected length matches
    with the detector size in the x-direction. Then we claim to
    have found an anode-cathode crossing muon.

    2. Anode- or cathode-piercing tracks:

    TODO describe here how it is done (similar to MicroBooNE).

    Then for any selected track, we split it into
    segments using a cubic 3D binning of the detector volume.
    For each segment we perform PCA to estimate the direction,
    hence dx, and we sum reco charge to get dQ.


    This heuristic relies only on semantic segmentation and
    particle clustering predictions.

    Note: we might need to use PID to weed out pion tracks.
    Note: we might exclude muons attached to a Michel for
    anode-piercing or cathode-piercing tracks.

    Configuration
    =============
    Under `processor_cfg`, you can specify the following parameters:

    spatial_size
    selection_threshold
    bin_size
    mode
    data
    vdrift
    drift
    voxel_size
    invert_regions
    limits
    volume_thresholds
    Michel_threshold
    pid_constraint
    muon_label
    track_label

    Output
    ======
    Two CSV files should contain respectively:
    - array of measurements of dQ/dx for each non-empty cell (`acpt_muons_cells.csv`)
    - array of detected muons and related information (`acpt_muons.csv`)
    """
    #
    # ====== Configuration ======
    #
    muon_cells, muons = [], []
    deghosting = analysis_cfg['analysis']['deghosting']

    processor_cfg       = analysis_cfg['analysis'].get('processor_cfg', {})
    spatial_size        = processor_cfg['spatial_size']
    selection_threshold = processor_cfg['selection_threshold']
    bin_size            = processor_cfg['bin_size']
    # 'ac' for anode-cathode piercing tracks
    mode           = processor_cfg.get('mode', ['ac'])
    # Whether we are running on MC or data
    data           = processor_cfg.get('data', False)
    # drift velocity of electrons in liquid argon in cm / us
    vdrift         = processor_cfg.get('vdrift', 0.1114)
    # drift the full length of TPC in cm
    drift_length   = processor_cfg.get('drift', 150)
    # Voxel size in cm
    voxel_size     = processor_cfg.get('voxel_size', 0.3)
    # Regions along x-direction where we need to reverse anode/cathode, ie
    # these regions go cathode -> anode in increasing x
    # expecting [[xmin, xmax]] format in px
    invert_regions = processor_cfg.get('invert_regions', [])
    # YZ plane detector limits - expecting a list of [min_y, max_y, min_z, max_z]
    limits         = processor_cfg.get('limits', [])
    if len(limits) == 0:
        min_y, min_z = 0, 0
        max_y, max_z = spatial_size, spatial_size
    elif len(limits) != 4:
        raise Exception('Limits ill-defined, expected a list of [min_y, max_y, min_z, max_z]')
    else:
        min_y, max_y, min_z, max_z = limits
    # Define "active" volume in YZ for this selection for anode- or cathode-piercing tracks
    # expecting [bottom_threshold, top_threshold, back_threshold, front_threshold]
    volume_thresholds = processor_cfg.get('volume_thresholds', [])
    if len(volume_thresholds) == 0:
        bottom_threshold, top_threshold, back_threshold, front_threshold = selection_threshold, selection_threshold, selection_threshold, selection_threshold
    elif len(volume_thresholds) != 4:
        raise Exception('Volume thresholds in YZ plane ill-defined, expected a list [bottom_threshold, top_threshold, back_threshold, front_threshold]')
    else:
        bottom_threshold, top_threshold, back_threshold, front_threshold = volume_thresholds
    # Whether to exclude tracks that are close to Michel voxels
    # threshold =-1 to disable, otherwise it is the threshold below which we consider the track
    # might be attached to a Michel electron.
    Michel_threshold = processor_cfg.get('Michel_threshold', -1)
    # Whether to enforce PID constraint (predicted as muon only)
    pid_constraint   = processor_cfg.get('pid_constraint', False)
    # Avoid hardcoding labels
    muon_label       = processor_cfg.get('muon_label', 2)
    track_label      = processor_cfg.get('track_label', 1)
    # To correct x coordinates - one cathode per cryostat, in same order as volumes
    cathode_x        = processor_cfg.get('cathode_x', [675.5, 2077])

    #
    # ====== Initialization ======
    #
    # Initialize analysis differently depending on data/MC setting
    if not data:
        predictor = FullChainEvaluator(data_blob, res, cfg, processor_cfg, deghosting=deghosting)
    else:
        predictor = FullChainPredictor(data_blob, res, cfg, processor_cfg, deghosting=deghosting)

    image_idxs = data_blob['index']

    # x is drift direction
    # y is vertical direction
    # z is beam direction
    x, y, z = 0, 1, 2

    pca = PCA(n_components=2)

    for i, index in enumerate(image_idxs):
        index_dict = {
            'Index': index,
            'run': data_blob['run_info'][i][0],
            'subrun': data_blob['run_info'][i][1],
            'event': data_blob['run_info'][i][2]
        }
        print(index_dict)
        pred_particles = predictor.get_particles(i, only_primaries=False)

        # Match with true particles if available
        if not data:
            true_particles = predictor.get_true_particles(i, only_primaries=False)
            # Match true particles to predicted particles
            true_ids = np.array([p.id for p in true_particles])
            matched_particles = predictor.match_particles(i, mode='true_to_pred', min_overlap=0.1)

        # Loop over predicted particles
        for p in pred_particles:
            if p.semantic_type != track_label: continue
            # We found a predicted track particle, examine if it is piercing
            coords = p.points
            projected_x_length = coords[:, x].max() - coords[:, x].min()
            touching_top = max_y - coords[:, y].max() < top_threshold
            touching_bottom = coords[:, y].min() - min_y < bottom_threshold
            touching_back = coords[:, z].min() - min_z < back_threshold
            touching_front = max_z - coords[:, z].max() < front_threshold

            # decide whether this track is worth keeping depending on mode
            keep = {'ac': False, 'a': False, 'c': False}
            keep['ac'] = (projected_x_length > (drift_length/voxel_size) - selection_threshold)
            piercing_x = -1
            if (touching_top ^ touching_bottom ^ touching_back ^ touching_front):
                # Assuming downward going
                entering, exiting = None, None
                if touching_top:
                    entering = coords[coords[:, y].argmax()]
                    exiting = coords[coords[:, y].argmin()]
                    keep['a'] = exiting[x] < entering[x]
                    keep['c'] = not keep['a']
                elif touching_bottom:
                    entering = coords[coords[:, y].argmin()]
                    exiting = coords[coords[:, y].argmax()]
                    keep['a'] = exiting[x] > entering[x]
                    keep['c'] = not keep['a']
                elif touching_back:
                    # Not really always "entering" - just smallest z.
                    entering = coords[coords[:, z].argmin()]
                    # Not really always "exiting" - just largest z.
                    exiting = coords[coords[:, z].argmax()]
                    keep['a'] = entering[x] > exiting[x]
                    keep['c'] = not keep['a']
                elif touching_front:
                    entering = coords[coords[:, z].argmax()]
                    exiting = coords[coords[:, z].argmin()]
                    keep['a'] = exiting[x] < entering[x]
                    keep['c'] = not keep['a']
                piercing_x = min(entering[x], exiting[x]) if keep['a'] else max(entering[x], exiting[x])
                # Invert if necessary
                for r in invert_regions:
                    if (entering[x] > r[0] and entering[x] < r[1]) or (exiting[x] > r[0] and exiting[x] < r[1]):
                        temp = keep['c']
                        keep['c'] = keep['a']
                        keep['a'] = temp
                        break
            # FIXME does an ACPT count as an APT or CPT ?
            # if keep['ac']:
            #     keep['a'] = True
            #     keep['c'] = True

            # If asked, check for presence of Michel electron
            if Michel_threshold >= 0:
                for p2 in pred_particles:
                    if p2.semantic_type != 2: continue
                    if cdist(p.points, p2.points).min() >= Michel_threshold: continue
                    keep['ac'] = False
                    keep['a'] = False
                    keep['c'] = False

            # If asked to check predicted PID, exclude non-predicted-muons
            if pid_constraint and p.pid != muon_label: continue

            if np.any([(keep[key] and key in mode) for key in keep]):
                # We are keeping this candidate, see if it is a match
                # for the records keeping

                # T0 correction, assuming T0 is minimal drift coordinate
                # keep in mind vdrift is in cm / us, so t0 is in us
                inverted = False
                if keep['ac']:
                    inverted = must_invert(coords[:, x].min(), invert_regions)
                    if inverted:
                        piercing_x = coords[:, x].min()
                        #t0 = piercing_x * voxel_size / vdrift
                    else:
                        piercing_x = coords[:, x].max()
                        #t0 = - piercing_x * voxel_size / vdrift
                    t0 = -1
                elif keep['a']: # FIXME account for invert_regions
                    t0 = piercing_x * voxel_size / vdrift
                elif keep['c']: # FIXME account for invert_regions
                    t0 = piercing_x * voxel_size / vdrift - drift_length/vdrift

                # don't forget coords are in voxel units
                coords[:, x] = coords[:, x] + (cathode_x[p.volume] - piercing_x)
                theta_yz = np.arctan2((coords[:, y].max() - coords[:, y].min()),(coords[:, z].max()-coords[:, z].min()))
                theta_xz = np.arctan2((coords[:, x].max() - coords[:, x].min()),(coords[:, z].max()-coords[:, z].min()))
                update_dict = {
                    'index': index,
                    'pred_particle_type': p.pid,
                    'pred_particle_is_primary': p.is_primary,
                    'pred_particle_size': p.size,
                    'projected_x_length': projected_x_length,
                    'theta_yz': theta_yz,
                    'theta_xz': theta_xz,
                    'coords_min_x': coords[:, x].min(),
                    'coords_max_x': coords[:, x].max(),
                    'piercing_x': piercing_x,
                    'must_invert': inverted,
                    'matched': len(p.match),
                    't0': t0,
                    'true_pdg': -1,
                    'true_size': -1,
                    'volume': p.volume
                    }
                update_dict.update(keep) # Record what kind of piercing track it was
                update_dict.update(index_dict)
                if not data and len(p.match)>0:
                    m = np.array(true_particles)[true_ids == p.match[0]][0]
                    update_dict.update({
                        'true_pdg': m.pid,
                        'true_size': m.size
                    })
                muons.append(OrderedDict(update_dict))
                track_dict = update_dict

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
                                'cell_dQ': p.depositions[cell].sum(),
                                'cell_dN': np.count_nonzero(cell),
                                'cell_dx': coords_pca.max() - coords_pca.min(),
                                'cell_deltax': coords[cell][:, x].max() - coords[cell][:, x].min(),
                                'cell_deltay': coords[cell][:, y].max() - coords[cell][:, y].min(),
                                'cell_deltaz': coords[cell][:, z].max() - coords[cell][:, z].min(),
                                'cell_ybin': y_idx,
                                'cell_zbin': z_idx,
                                'cell_xbin': x_idx,
                                'volume': p.volume,
                            })
                            # Include parent track information
                            update_dict.update(track_dict)
                            update_dict.update(index_dict)
                            muon_cells.append(update_dict)

    return [muon_cells, muons]
