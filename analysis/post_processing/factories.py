from . import reconstruction, pmt, crt, trigger, evaluation

POST_PROCESSOR_DICT = {}
for module in [reconstruction, pmt, crt, trigger, evaluation]:
    for processor in dir(module):
        if 'Processor' in processor:
            cls = getattr(module, processor)
            POST_PROCESSOR_DICT[cls.name] = cls


def post_processor_factory(name, cfg, parent_path=''):
    '''
    Instantiates post-processor based on name specified in configuration under
    the `post_processor` config block.

    Parameters
    ----------
    name : str
        Name of the post-processor
    cfg : dict
        Configuration dictionary
    parent_path : str
        Path to the parent directory of the main analysis configuration. This
        allows for the use of relative paths

    Returns
    -------
    object
         Initialized post-processor object
    '''
    # Check that the post processor is known
    if name not in POST_PROCESSOR_DICT:
        raise KeyError(f'Post-processor name not recognized: {name}')

    # Set the parent path
    POST_PROCESSOR_DICT[name].parent_path = parent_path

    # Initialize
    return POST_PROCESSOR_DICT[name](**cfg)

