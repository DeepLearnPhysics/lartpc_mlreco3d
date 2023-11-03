from copy import deepcopy

RULES = {
    'parse_sparse2d': ['tensor', None],
    'parse_sparse3d': ['tensor', None, False, True],
    'parse_sparse3d_ghost': ['tensor', None, False, True],
    'parse_sparse3d_charge_rescaled': ['tensor', None, False, True],

    'parse_cluster2d': ['tensor', None],
    'parse_cluster3d': ['tensor', None, False, True],
    'parse_cluster3d_charge_rescaled': ['tensor', None, False, True],
    'parse_cluster3d_2cryos': ['tensor', None, False, True],

    'parse_particles': ['list'],
    'parse_neutrinos': ['list'],
    'parse_particle_points': ['tensor', None, False, True],
    'parse_particle_coords': ['tensor', None, False, True],
    'parse_particle_graph': ['tensor', None],
    'parse_particle_singlep_pdg': ['tensor', None],
    'parse_particle_singlep_einit': ['tensor', None],

    'parse_meta2d': ['list'],
    'parse_meta3d': ['list'],
    'parse_run_info': ['list'],
    'parse_opflash': ['list'],
    'parse_crthits': ['list'],
    'parse_trigger': ['list']
}

def input_unwrap_rules(schemas):
    '''
    Translates parser schemas into unwrap rules.

    Parameters
    ----------
    schemas : dict
        Dictionary of parser schemas

    Returns
    -------
    dict
        Dictionary of unwrapping rules
    '''
    rules = {}
    for name, schema in schemas.items():
        parser = schema['parser']
        assert parser in RULES, f'Unable to unwrap data from {parser}'
        rules[name] = deepcopy(RULES[parser])
        if rules[name][0] == 'tensor':
            rules[name][1] = name

    return rules

