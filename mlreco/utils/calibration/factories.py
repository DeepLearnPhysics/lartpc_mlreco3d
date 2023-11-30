from . import gain, lifetime, transparency, field, recombination

CALIBRATOR_DICT = {}
for module in [gain, lifetime, transparency, recombination, field]:
    for calibrator in dir(module):
        if 'Calibrator' in calibrator:
            cls = getattr(module, calibrator)
            CALIBRATOR_DICT[cls.name] = cls


def calibrator_factory(name, cfg, parent_path=''):
    '''
    Instantiates calibrator based on name specified in configuration under
    the `calibrator` config block.

    Parameters
    ----------
    name : str
        Name of the calibrator
    cfg : dict
        Configuration dictionary
    parent_path : str
        Path to the parent directory of the main analysis configuration. This
        allows for the use of relative paths

    Returns
    -------
    object
         Initialized calibrator object
    '''
    # Check that the calibrator is known
    if name not in CALIBRATOR_DICT:
        raise KeyError(f'Calibrator name not recognized: {name}')

    # Set the parent path
    CALIBRATOR_DICT[name].parent_path = parent_path

    # Initialize
    return CALIBRATOR_DICT[name](**cfg)

