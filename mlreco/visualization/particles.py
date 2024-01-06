import numpy as np

from .points import scatter_points
from .plotly_layouts import HIGH_CONTRAST_COLORS

from mlreco.utils.particles import get_interaction_ids, get_nu_ids, get_particle_ids, get_shower_primary_ids, get_group_primary_ids
from mlreco.utils.globals import COORD_COLS, PART_COL

def scatter_particles(cluster_label, particles, particles_mpv=None, neutrinos=None, part_col=PART_COL, markersize=1, **kwargs):
    '''
    Function which returns a graph object per true particle in the 
    particle list, provided that the particle deposited energy in the
    detector which appears in the cluster_label tensor.

    Parameters
    ----------
    cluster_label : np.ndarray
        (N, M) Tensor of pixel coordinates and their associated cluster ID
    particles : List[larcv.Particle]
        (P) List of LArCV true particle objects
    particles_mpv : List[larcv.Particle], optional
        (M) List of true MPV particle instances
    neutrinos : List[larcv.Neutrino], optional
        (N) List of true neutrino instances
    part_col : int
        Index of the column in the label tensor that contains the particle ID
    **kwargs : dict, optional
        List of additional arguments to pass to plotly.graph_objs.Scatter3D that
        make up the output list

    Returns
    -------
    List[plotly.graph_objs.Scatter3D]
        List of particle traces
    '''
    # Get the labels that are not immediately provided by the larcv.Particle objects
    inter_ids = get_interaction_ids(particles)
    nu_ids    = get_nu_ids(particles, inter_ids, particles_mpv, neutrinos)
    pid_ids   = get_particle_ids(particles, nu_ids, include_mpr=True, include_secondary=True)
    pshow_ids = get_shower_primary_ids(particles)
    pgrp_ids  = get_group_primary_ids(particles, nu_ids, include_mpr=True)

    # Initialize one graph per particle
    traces = []
    colors = HIGH_CONTRAST_COLORS
    for i in range(len(particles)):
        # Get a mask that corresponds to the particle entry, skip empty particles
        mask = cluster_label[:, part_col] == i
        if not np.sum(mask): continue
            
        # Initialize the information string
        p = particles[i]
        start = p.first_step().x(), p.first_step().y(), p.first_step().z()
        position = p.x(), p.y(), p.z()
        anc_start = p.ancestor_x(), p.ancestor_y(), p.ancestor_z()
        
        label = f'Particle {p.id()}'
        hovertext_dict = {'Particle ID': p.id(),
                          'Group ID': p.group_id(),
                          'Parent ID': p.parent_id(),
                          'Inter. ID': inter_ids[i],
                          'Neutrino ID': nu_ids[i],
                          'Type ID': pid_ids[i],
                          'Shower primary': pshow_ids[i],
                          'Inter. primary': pgrp_ids[i],
                          'Shape ID': p.shape(),
                          'PDG code': p.pdg_code(),
                          'Parent PDG code': p.parent_pdg_code(),
                          'Anc. PDG code': p.ancestor_pdg_code(),
                          'Process': p.creation_process(),
                          'Parent process': p.parent_creation_process(),
                          'Anc. process': p.ancestor_creation_process(),
                          'Initial E': f'{p.energy_init():0.1f} MeV',
                          'Deposited E': f'{p.energy_deposit():0.1f} MeV',
                          'Position': f'({position[0]:0.3e}, {position[1]:0.3e}, {position[2]:0.3e})',
                          'Start point': f'({start[0]:0.3e}, {start[1]:0.3e}, {start[2]:0.3e})',
                          'Anc. start point': f'({anc_start[0]:0.3e}, {anc_start[1]:0.3e}, {anc_start[2]:0.3e})'}

        hovertext = ''.join([f'{l}:   {v}<br>' for l, v in hovertext_dict.items()])
        
        # Append a scatter plot trace
        trace = scatter_points(cluster_label[mask][:, COORD_COLS], color=str(colors[i%len(colors)]), hovertext=hovertext, markersize=markersize, **kwargs)
        trace[0]['name'] = label
        
        traces += trace
        
    return traces
