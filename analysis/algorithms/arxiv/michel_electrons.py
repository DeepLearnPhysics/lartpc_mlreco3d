from collections import OrderedDict
from turtle import update
from sklearn.decomposition import PCA

from analysis.classes.predictor import FullChainPredictor
from analysis.classes.evaluator import FullChainEvaluator
from lartpc_mlreco3d.analysis.algorithms.arxiv.decorator import evaluate
from lartpc_mlreco3d.analysis.algorithms.arxiv.calorimetry import compute_track_length

from pprint import pprint
import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import pandas as pd


def yz_calibrations(yz_calib, points, edep, bin_size=10, voxel_size=0.3, spatial_size=6144):
    """
    Apply YZ calibration factors from CSV file.

    Parameters
    ==========
    yz_calib: pd.DataFrame
    points: (N, 3) np.ndarray
    edep: (N,) np.ndarray
    bin_size: float, in cm
    voxel_size: float, in cm
    spatial_size: int, in voxels

    Returns
    =======
    float
        calibrated sum of edep
    """
    if yz_calib is None:
        return edep.sum()

    xbins = np.arange(0, spatial_size, bin_size/voxel_size)
    ybins = np.arange(0, spatial_size, bin_size/voxel_size)
    zbins = np.arange(0, spatial_size, bin_size/voxel_size)
    xidx = np.digitize(points[:, 0], xbins)
    yidx = np.digitize(points[:, 1], ybins)
    zidx = np.digitize(points[:, 2], zbins)
    df = pd.DataFrame({'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2], 'edep': edep})
    df['xbin'] = xidx
    df['ybin'] = yidx
    df['zbin'] = zidx
    df['tpc'] = 'EE'
    df.loc[(df['xbin'] > 40) & (df['xbin'] <= 83), 'tpc'] = 'EW'
    df.loc[(df['xbin'] > 83) & (df['xbin'] <= 125), 'tpc'] = 'WE'
    df.loc[df['xbin'] > 125, 'tpc'] = 'WW'
    df = pd.merge(df, yz_calib, on=['tpc', 'ybin', 'zbin'], how='left')
    df.loc[np.isnan(df['scale']), 'scale'] = 1.
    return (df['edep'] * df['scale']).sum()

def get_bounding_box(points):
    return {
            "points_min_x": points[:, 0].min(),
            "points_max_x": points[:, 0].max(),
            "points_min_y": points[:, 1].min(),
            "points_max_y": points[:, 1].max(),
            "points_min_z": points[:, 2].min(),
            "points_max_z": points[:, 2].max()
    }

def is_attached_at_edge(points1, points2, attached_threshold=5,
                        one_pixel=5, ablation_radius=15, ablation_min_samples=5,
                        return_dbscan_cluster_count=False):
    distances = cdist(points1, points2)
    is_attached = np.min(distances) < attached_threshold
    # check for the edge now
    Michel_min, MIP_min = np.unravel_index(np.argmin(distances), distances.shape)
    min_coords = points2[MIP_min, :]
    ablated_cluster = points2[np.linalg.norm(points2-min_coords, axis=1) > ablation_radius]
    new_cluster_count, old_cluster_count = 0, 1
    if ablated_cluster.shape[0] > 0:
        dbscan = DBSCAN(eps=one_pixel, min_samples=ablation_min_samples)
        old_cluster = dbscan.fit(points2).labels_
        new_cluster = dbscan.fit(ablated_cluster).labels_
        # If only one cluster is left, we were at the edge
        # Account for old cluster count in case track is fragmented
        # and put together by Track GNN
        old_cluster_count = len(np.unique(old_cluster[old_cluster>-1]))
        new_cluster_count =  len(np.unique(new_cluster[new_cluster>-1]))
        is_edge = (old_cluster_count - new_cluster_count) <= 1 and old_cluster_count >= new_cluster_count
    else: # if nothing is left after ablating, this is a really small muon... calling it the edge
        is_edge = True

    if return_dbscan_cluster_count:
        return is_attached and is_edge, new_cluster_count, old_cluster_count

    return is_attached and is_edge

def find_cosmic_angle(muon, michel, endpoint, radius=30):
    """
    Parameters
    ==========
    muon: Particle
    michel: Particle
    """
    # Find muon end direction
    pca = PCA(n_components=2)
    neighborhood = (cdist(muon.points, [endpoint]) < radius).reshape((-1,))
    if neighborhood.sum() < 3:
        return -1 * np.ones((3,)), -1. * np.ones((3,))
    coords = pca.fit_transform(muon.points[neighborhood])
    muon_dir = pca.components_[0, :]
    # Make sure muon direction is in the right sense
    far_point = muon.points[neighborhood][cdist(muon.points[neighborhood], [endpoint]).argmax()]
    vec = endpoint - far_point
    if np.dot(vec, muon_dir) < 0:
        muon_dir *= -1
    # Find Michel start direction
    pca = PCA(n_components=2)
    neighborhood = (cdist(michel.points, [endpoint]) < radius).reshape((-1,))
    if neighborhood.sum() < 3:
        return -1 * np.ones((3,)), -1. * np.ones((3,))
    coords = pca.fit_transform(michel.points[neighborhood])
    michel_dir = pca.components_[0, :]
    # Make sure Michel direction is in the right sense
    far_point = michel.points[neighborhood][cdist(michel.points[neighborhood], [endpoint]).argmax()]
    vec = far_point - endpoint
    if np.dot(vec, michel_dir) < 0:
        michel_dir *= -1

    return muon_dir, michel_dir

def find_true_cosmic_angle(muon, michel, particles_asis_voxels, radius=30):
    #true_michel, true_muon = None, None
    #for p in particles_asis_voxels:
    #    if p.id() == muon.id:
    #        true_muon = p
    #    if p.id() == michel.id:
    #        true_michel = p
    if michel is None or muon is None:
        return -1 * np.ones((3,)), -1 * np.ones((3,))
    #endpoint = [true_michel.x(), true_michel.y(), true_michel.z()]
    #endpoint = [true_muon.end_position().x(), true_muon.end_position().y(), true_muon.end_position().z()]
    endpoint_id = cdist(michel.points, muon.points).argmin()
    michel_id, muon_id = np.unravel_index(endpoint_id, (len(michel.points), len(muon.points)))
    endpoint = muon.points[muon_id]
    return find_cosmic_angle(muon, michel, endpoint, radius=radius)

@evaluate(['michels_pred', 'michels_true'])
def michel_electrons(data_blob, res, data_idx, analysis_cfg, cfg):
    """
    Selection of Michel electrons
    =============================


    Configuration
    =============
    Under `processor_cfg`, you can specify the following parameters:


    Output
    ======
    """
    try:
        yz_calib = pd.read_csv("/sdf/group/neutrino/ldomine/ICARUS_Calibrations/v09_62_00/tpc_yz_correction_data.csv")
    except Exception as e:
        print("Unable to load YZ calibration csv.")
        print(e)
        yz_calib = None
    #
    # ====== Configuration ======
    #
    michels, true_michels = [], []
    deghosting = analysis_cfg['analysis']['deghosting']

    processor_cfg       = analysis_cfg['analysis'].get('processor_cfg', {})
    spatial_size        = processor_cfg['spatial_size']
    # Whether we are running on MC or data
    data           = processor_cfg.get('data', False)

    # Avoid hardcoding labels
    michel_label       = processor_cfg.get('michel_label', 2)
    track_label        = processor_cfg.get('track_label', 1)
    muon_label         = processor_cfg.get('muon_label', 2)
    pion_label         = processor_cfg.get('pion_label', 3)
    shower_label       = processor_cfg.get('shower_label', 0)

    # Thresholds
    attached_threshold = processor_cfg.get('attached_threshold', 5)
    one_pixel          = processor_cfg.get('ablation_eps', 5)
    ablation_radius    = processor_cfg.get('ablation_radius', 15)
    ablation_min_samples = processor_cfg.get('ablation_min_samples', 5)
    shower_threshold   = processor_cfg.get('shower_threshold', 10)
    fiducial_threshold = processor_cfg.get('fiducial_threshold', 30)
    muon_min_voxel_count = processor_cfg.get('muon_min_voxel_count', 30)
    matching_mode      = processor_cfg.get('matching_mode', 'true_to_pred')
    #
    # ====== Initialization ======
    #
    # Initialize analysis differently depending on data/MC setting
    start = time.time()
    if not data:
        predictor = FullChainEvaluator(data_blob, res, cfg, processor_cfg, deghosting=deghosting)
    else:
        predictor = FullChainPredictor(data_blob, res, cfg, processor_cfg, deghosting=deghosting)
    print("Evaluator took %d s" % (time.time() - start))
    image_idxs = data_blob['index']


    start = time.time()
    for i, index in enumerate(image_idxs):
        index_dict = {
            'Index': index,
            'run': data_blob['run_info'][i][0],
            'subrun': data_blob['run_info'][i][1],
            'event': data_blob['run_info'][i][2]
        }
        pred_particles = predictor.get_particles(i, only_primaries=False)

        # Match with true particles if available
        if not data:
            true_particles = predictor.get_true_particles(i, only_primaries=False)
            # Match true particles to predicted particles
            true_ids = np.array([p.id for p in true_particles])
            matched_particles = predictor.match_particles(i, mode=matching_mode, min_overlap=5, overlap_mode='counts')

            for tp in true_particles:
                if tp.semantic_type != michel_label: continue
                #if not predictor.is_contained(tp.points, threshold=fiducial_threshold): continue
                p = data_blob['particles_asis'][i][tp.id]

                michel_is_attached_at_edge = False
                distance_to_muon = np.infty
                muon_ablation_cluster_count = -1
                muon_ablation_delta = None
                for tp2 in true_particles:
                    if tp2.semantic_type != track_label: continue
                    if tp2.pid != muon_label and tp2.pid != pion_label: continue
                    if tp2.size < muon_min_voxel_count: continue
                    d = cdist(tp.points, tp2.points).min()
                    attached_at_edge, cluster_count, old_cluster_count = is_attached_at_edge(tp.points, tp2.points,
                                                                    attached_threshold=attached_threshold,
                                                                    one_pixel=one_pixel,
                                                                    ablation_radius=ablation_radius,
                                                                    ablation_min_samples=ablation_min_samples,
                                                                    return_dbscan_cluster_count=True)
                    if d < distance_to_muon:
                        distance_to_muon = d
                        muon_ablation_cluster_count = cluster_count
                        muon_ablation_delta = old_cluster_count - cluster_count
                    if not attached_at_edge: continue
                    michel_is_attached_at_edge = True
                    break

                # Determine whether it was matched
                is_matched = False
                for mp in matched_particles: # matching is done true to pred
                    true_idx = 0 if matching_mode == 'true_to_pred' else 1
                    if mp[true_idx] is None or mp[true_idx].id != tp.id or mp[true_idx].volume != tp.volume: continue
                    is_matched = True
                    break

                # DBSCAN evaluation
                dbscan = DBSCAN(eps=one_pixel, min_samples=ablation_min_samples)
                michel_clusters = dbscan.fit(tp.points).labels_
                # Record true Michel
                true_michels.append(OrderedDict({
                    'index': index,
                    'true_num_pix': tp.size,
                    'true_sum_pix': tp.depositions.sum(),
                    'is_attached_at_edge': michel_is_attached_at_edge,
                    'muon_ablation_cluster_count': muon_ablation_cluster_count,
                    'muon_ablation_delta': muon_ablation_delta,
                    'is_matched': is_matched,
                    'energy_init': p.energy_init(),
                    'energy_deposit': p.energy_deposit(),
                    'num_voxels': p.num_voxels(),
                    'distance_to_muon': distance_to_muon,
                    'michel_cluster_count': len(np.unique(michel_clusters[michel_clusters>-1])),
                    'volume': tp.volume
                }))
                true_michels[-1].update(get_bounding_box(tp.points))
                true_michels[-1].update(index_dict)

        # Loop over predicted particles
        for p in pred_particles:
            if p.semantic_type != michel_label: continue
            print("Found a predicted Michel particle!")
            #if not predictor.is_contained(p.points, threshold=fiducial_threshold): continue
            # Check whether it is attached to the edge of a track
            michel_is_attached_at_edge = False
            muon = None
            for p2 in pred_particles:
                if p2.semantic_type != track_label: continue
                if p2.size < muon_min_voxel_count: continue
                if not is_attached_at_edge(p.points, p2.points,
                                        attached_threshold=attached_threshold,
                                        one_pixel=one_pixel,
                                        ablation_radius=ablation_radius,
                                        ablation_min_samples=ablation_min_samples): continue
                michel_is_attached_at_edge = True
                muon = p2
                break

            # Require that the Michel is attached at the edge of a track
            if not michel_is_attached_at_edge: continue
            print('... is attached at edge of a muon')

            # Look for shower-like particles nearby
            daughter_showers = []
            distance_to_closest_shower = np.infty
            if shower_threshold > -1:
                for p2 in pred_particles:
                    if p2.id == p.id: continue
                    if p2.semantic_type != shower_label and p2.semantic_type != michel_label: continue
                    distance_to_closest_shower = min(distance_to_closest_shower, cdist(p.points, p2.points).min())
                    if distance_to_closest_shower >= shower_threshold: continue
                    daughter_showers.append(p2)
            daughter_num_pix = np.sum([daugher.size for daugher in daughter_showers])
            daughter_sum_pix = np.sum([daughter.depositions.sum() for daughter in daughter_showers])

            # Record muon endpoint
            endpoint = np.array([-1, -1, -1])
            if cdist(p.points, [muon.startpoint]).min() > cdist(p.points, [muon.endpoint]).min():
                if (muon.endpoint != np.array([-1, -1, -1])).any():
                    endpoint = muon.endpoint
            else:
                if (muon.startpoint != np.array([-1, -1, -1])).any():
                    endpoint = muon.startpoint
            #if (endpoint == np.array([-1, -1, -1])).all():
            endpoint_id = cdist(p.points, muon.points).argmin()
            michel_id, muon_id = np.unravel_index(endpoint_id, (len(p.points), len(muon.points)))
            endpoint = muon.points[muon_id]

            # Find angle between Michel and muon
            muon_dir, michel_dir = find_cosmic_angle(muon, p, endpoint)

            # Heuristic to isolate primary ionization
            dbscan = DBSCAN(eps=one_pixel, min_samples=ablation_min_samples)
            clabels = dbscan.fit(p.points).labels_
            pionization = clabels[michel_id] # cluster label of Michel point closest to the muon
            primary_ionization = clabels == pionization

            # Record distance to muon
            michel_to_muon_distance = cdist(p.points, muon.points).min()

            # Record calibrated depositions sum
            pred_sum_pix_calib = yz_calibrations(yz_calib, p.points, p.depositions)
            pred_primary_sum_pix_calib = yz_calibrations(yz_calib, p.points[primary_ionization], p.depositions[primary_ionization])
            print("caibrating ", p.depositions.sum(), pred_sum_pix_calib)

            # Record candidate Michel
            update_dict = {
                'index': index,
                'particle_id': p.id,
                'muon_id': muon.id,
                'volume': p.volume,
                'muon_size': muon.size,
                'daughter_num_pix': daughter_num_pix,
                'daughter_sum_pix': daughter_sum_pix,
                'distance_to_closest_shower': distance_to_closest_shower,
                'pred_num_pix': p.size,
                'pred_sum_pix': p.depositions.sum(),
                'true_num_pix': -1,
                'true_sum_pix': -1,
                'true_primary_num_pix': -1,
                'true_primary_sum_pix': -1,
                'pred_num_pix_true': -1,
                'pred_sum_pix_true': -1,
                'michel_true_energy_init': -1,
                'michel_true_energy_deposit': -1,
                'michel_true_num_voxels': -1,
                'cluster_purity': -1,
                'cluster_efficiency': -1,
                'matched': False,
                'true_pdg': -1,
                'true_id': -1,
                'true_semantic_type': -1,
                'true_noghost_num_pix': -1,
                'true_noghost_sum_pix': -1,
                'true_noghost_primary_num_pix': -1,
                'true_noghost_primary_sum_pix': -1,
                'endpoint_x': endpoint[0],
                'endpoint_y': endpoint[1],
                'endpoint_z': endpoint[2],
                'true_particle_track_id': -1,
                'true_particle_pdg': -1,
                'true_particle_parent_pdg': -1,
                'true_particle_energy_init': -1,
                'true_particle_energy_deposit': -1,
                'true_particle_px': -1,
                'true_particle_py': -1,
                'true_particle_pz': -1,
                'true_particle_t': -1,
                'true_particle_parent_t': -1,
                'muon_dir_x': muon_dir[0],
                'muon_dir_y': muon_dir[1],
                'muon_dir_z': muon_dir[2],
                'michel_dir_x': michel_dir[0],
                'michel_dir_y': michel_dir[1],
                'michel_dir_z': michel_dir[2],
                'muon_true_dir_x': -1,
                'muon_true_dir_y': -1,
                'muon_true_dir_z': -1,
                'michel_true_dir_x': -1,
                'michel_true_dir_y': -1,
                'michel_true_dir_z': -1,
                'pred_primary_num_pix': primary_ionization.sum(),
                'pred_primary_sum_pix': p.depositions[primary_ionization].sum(),
                'true_primary_sum_pix_ADC': -1,
                'true_sum_pix_MeV': -1,
                'distance_to_muon': michel_to_muon_distance,
                'muon_length': compute_track_length(muon.points),
                'pred_sum_pix_calib': pred_sum_pix_calib,
                'true_sum_pix_calib': -1,
                'pred_primary_sum_pix_calib': pred_primary_sum_pix_calib,
                'true_primary_sum_pix_calib': -1
            }
            update_dict.update(index_dict)
            #print("Heuristic primary ", update_dict['pred_num_pix'], update_dict['pred_primary_num_pix'])

            if not data:
                for mp in matched_particles: # matching is done true2pred
                    true_idx = 0 if matching_mode == 'true_to_pred' else 1
                    pred_idx = 1 - true_idx
                    if mp[pred_idx] is None or mp[pred_idx].id != p.id or mp[pred_idx].volume != p.volume: continue
                    if mp[true_idx] is None: continue
                    if mp[true_idx].volume != p.volume: continue
                    m = mp[true_idx]
                    pe = m.purity_efficiency(p)
                    overlap_indices, mindices, _ = np.intersect1d(m.voxel_indices, p.voxel_indices, return_indices=True)
                    truep = data_blob['particles_asis'][i][m.id]
                    truemuon = None
                    if truep.parent_id() in true_ids:
                        truemuon = true_particles[np.where(true_ids == truep.parent_id())[0][0]]

                    cluster_label_noghost = predictor.get_true_label(i, "group", schema="cluster_label_noghost", volume=p.volume)
                    segment_label_noghost = predictor.get_true_label(i, "segment", schema="cluster_label_noghost", volume=p.volume)
                    charge_label_noghost = predictor.get_true_label(i, "charge", schema="cluster_label_noghost", volume=p.volume)

                    truecluster = cluster_label_noghost == m.id
                    trueprimary = (cluster_label_noghost == m.id) & (segment_label_noghost == michel_label)
                    muon_true_dir, michel_true_dir = find_true_cosmic_angle(truemuon, m, data_blob['particles_asis_voxels'][i])

                    #input_data = predictor.get_true_label(i, "charge", schema="input_data_rescaled", volume=p.volume)
                    input_data = res['input_rescaled'][i*2+p.volume][:, 4]
                    cluster_label_pred_noghost = predictor.get_true_label(i, "group", schema="cluster_label", volume=p.volume)
                    segment_label_pred_noghost = predictor.get_true_label(i, "segment", schema="cluster_label", volume=p.volume)
                    charge_label_pred_noghost = predictor.get_true_label(i, "charge", schema="cluster_label", volume=p.volume)
                    truecluster_pred = cluster_label_pred_noghost == m.id
                    trueprimary_pred = truecluster_pred & (segment_label_pred_noghost == michel_label)

                    # Calibrate too
                    true_sum_pix_calib = yz_calibrations(yz_calib, m.points, m.depositions)
                    true_primary_sum_pix_calib = yz_calibrations(yz_calib, res['input_rescaled'][i*2+p.volume][trueprimary_pred, 1:4], input_data[trueprimary_pred])
                    update_dict.update({
                        'matched': True,
                        'true_id': m.id,
                        'true_pdg': m.pid,
                        'true_particle_track_id': m.asis.track_id(),
                        'true_particle_pdg': m.asis.pdg_code(),
                        'true_particle_parent_pdg': m.asis.parent_pdg_code(),
                        'true_particle_energy_init': m.asis.energy_init(),
                        'true_particle_energy_deposit': m.asis.energy_deposit(),
                        'true_particle_px': m.asis.px(),
                        'true_particle_py': m.asis.py(),
                        'true_particle_pz': m.asis.pz(),
                        'true_particle_t': m.asis.t(),
                        'true_particle_parent_t': m.asis.parent_t(),
                        'true_semantic_type': m.semantic_type,
                        'cluster_purity': pe['purity'],
                        'cluster_efficiency': pe['efficiency'],
                        # true_* uses the entire Michel if michel_primary_ionization_only if False
                        # but otherwise, only primary ionization
                        # using predicted deghosting mask
                        'true_num_pix': m.size,
                        'true_sum_pix': m.depositions.sum(), # in ADC
                        'true_sum_pix_MeV': charge_label_pred_noghost[truecluster_pred].sum(), # in MeV
                        'true_primary_num_pix': trueprimary_pred.sum(),
                        'true_primary_sum_pix': charge_label_pred_noghost[trueprimary_pred].sum(), # will be in MeV
                        'true_primary_sum_pix_ADC': input_data[trueprimary_pred].sum(), # in ADC
                        'pred_num_pix_true': len(overlap_indices),
                        'pred_sum_pix_true': m.depositions[mindices].sum(),
                        'michel_true_energy_init': truep.energy_init(),
                        'michel_true_energy_deposit': truep.energy_deposit(),
                        'michel_true_num_voxels': truep.num_voxels(),
                        # cluster_label includes radiative photon in Michel true particle
                        # so using segmentation labels to find primary ionization as well
                        # voxel sum will be in MeV here.
                        # true_noghost_* is using the true deghosting mask.
                        'true_noghost_num_pix': np.count_nonzero(truecluster),
                        'true_noghost_sum_pix': charge_label_noghost[truecluster].sum(), # in MeV
                        'true_noghost_primary_num_pix': np.count_nonzero(trueprimary),
                        'true_noghost_primary_sum_pix': charge_label_noghost[trueprimary].sum(), # in MeV
                        'muon_true_dir_x': muon_true_dir[0],
                        'muon_true_dir_y': muon_true_dir[1],
                        'muon_true_dir_z': muon_true_dir[2],
                        'michel_true_dir_x': michel_true_dir[0],
                        'michel_true_dir_y': michel_true_dir[1],
                        'michel_true_dir_z': michel_true_dir[2],
                        'true_sum_pix_calib': true_sum_pix_calib,
                        'true_primary_sum_pix_calib': true_primary_sum_pix_calib
                    })
                    break

            update_dict.update(get_bounding_box(p.points))
            michels.append(OrderedDict(update_dict))
    print("Loop took %d s" % (time.time() - start))

    return [michels, true_michels]
