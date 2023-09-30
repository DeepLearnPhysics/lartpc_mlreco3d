import os
import numpy as np
import pandas as pd

from analysis.post_processing import post_processing

OPFLASH_ATTRS = np.array(['opflash', 'opflash_cryoE', 'opflash_cryoW'])


@post_processing(data_capture=['run_info'],
                 result_capture=[],
                 data_capture_optional=OPFLASH_ATTRS)
def parse_trigger(data_dict,
                  result_dict,
                  file_path,
                  correct_flash_times=True,
                  flash_time_corr_us=4):
    '''
    Parses trigger information from a CSV file and adds a new trigger_info
    data product to the data dictionary.

    Parameters
    ----------
    data_dict : dict
        Input data dictionary
    result_dict : dict
        Chain output dictionary
    file_path : str
        Path to the csv file which contains the trigger information
    correct_flash_times : bool, default True
        If True, corrects the flash times using w.r.t. the trigger times
    flash_time_corr_us : float, default 4
        Systematic correction between the trigger time and the flash time in us
    '''
    # Load the trigger information
    if not os.path.isfile(file_path):
        raise FileNotFoundError('Cannot find the trigger file')
    trigger_dict = pd.read_csv(file_path)

    # Fetch the run info, find the corresponding trigger, save attributes
    run_info = data_dict['run_info'][0] # TODO: Why? Get rid of index
    run_id, event_id = run_info['run'], run_info['event']
    trigger_mask = (trigger_dict['run_number'] == run_id) & \
            (trigger_dict['event_no'] == event_id)
    trigger_info = trigger_dict[trigger_mask]
    if not len(trigger_info):
        raise KeyError(f'Could not find run {run_id} event {event_id} in the trigger file')
    elif len(trigger_info) > 1:
        raise KeyError(f'Found more than one trigger associated with {run_id} event {event_id} in the trigger file')

    trigger_info = trigger_info.to_dict(orient='records')[0]
    del trigger_info['run_number'], trigger_info['event_no']

    # If requested, loop over the interaction objects, modify flash times
    if correct_flash_times:
        # Make sure there's at least one optical flash attribute
        mask = np.array([attr in data_dict for attr in OPFLASH_ATTRS])
        assert mask.any(), 'Did not find optical flashes to correct the time of'

        # Loop over flashes, correct the timing (flash times are in us)
        offset = (trigger_info['wr_seconds']-trigger_info['beam_seconds'])*1e6 \
                + (trigger_info['wr_nanoseconds']-trigger_info['beam_nanoseconds'])*1e-3 \
                - flash_time_corr_us
        for key in OPFLASH_ATTRS[mask]:
            for opflash in data_dict[key]:
                time = opflash.time()
                opflash.time(time + offset)

    return {'trigger_info': [trigger_info]}
