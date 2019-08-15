def input(csv_logger, data_blob, res, **kwargs):
    # 0 = Event voxels and values
    if 'input_data' in data_blob:
        for row in data_blob['input_data']:
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (row[0], row[1], row[2], 0, row[4]))
            csv_logger.write()
    # 1 = Labels for PPN
    if 'particles_label' in data_blob:
        for row in data_blob['particles_label']:
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (row[0], row[1], row[2], 1, row[4]))
            csv_logger.write()
    # 2 = UResNet labels
    if 'segment_label' in data_blob:
        for row in data_blob['segment_label']:
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (row[0], row[1], row[2], 2, row[4]))
            csv_logger.write()
    # clusters label
    # 15 = group id, 16 = semantic labels, 17 = energy
    if 'clusters_label' in data_blob:
        for row in data_blob['clusters_label']:
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (row[0], row[1], row[2], 15, row[5]))
            csv_logger.write()
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (row[0], row[1], row[2], 16, row[6]))
            csv_logger.write()
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (row[0], row[1], row[2], 17, row[4]))
            csv_logger.write()
    # 18 = cluster3d_mcst_true
    if 'cluster3d_mcst_true' in data_blob:
        for row in data_blob['cluster3d_mcst_true']:
            # csv_logger.record(('x', 'y', 'z', 'type', 'value'),
            #                   (row[0], row[1], row[2], 18, row[5]))
            # csv_logger.write()
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (row[0], row[1], row[2], 19, row[4]))
            csv_logger.write()
