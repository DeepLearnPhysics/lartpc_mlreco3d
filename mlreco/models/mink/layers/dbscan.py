import numpy as np
from mlreco.models.layers.dbscan import DBSCANFragmenter


class MinkDBSCANFragmenter(DBSCANFragmenter):
    """
    DBSCANFragmenter for ME Backend PPN
    """

    def __init__(self, cfg, name='dbscan_frag'):
        super(MinkDBSCANFragmenter, self).__init__(cfg, batch_col=0, coords_col=(1,4))
        self.batch_col = 0
        self.coords_col = (1, 4)

    def forward(self, data, output):

        from mlreco.utils.ppn import mink_ppn_selector

        # If tracks are clustered, get the track points from the PPN output
        data = data.detach().cpu().numpy()
        track_points = None
        if self.track_label in self.cluster_classes:
            # FIXME ppn_score not in output?
            numpy_output = {'segmentation': [output['segmentation'][0].detach().cpu().numpy()],
                            'points'      : [output['points'][0].detach().cpu().numpy()],
                            'mask_ppn'    : [output['mask_ppn'][0].detach().cpu().numpy()],
                            'ppn_score'   : [output['ppn_score'][0].detach().cpu().numpy()]}

            points =  mink_ppn_selector(data, numpy_output,
                                        score_threshold      = self.ppn_score_threshold,
                                        type_threshold       = self.ppn_type_threshold,
                                        type_score_threshold = self.ppn_type_score_threshold)
            point_labels = points[:,-1]
            track_points = points[(point_labels == self.track_label) | \
                                  (point_labels == self.michel_label),:self.dim+1]

        # Break down the input data to its components
        bids = np.unique(data[:, self.batch_col].astype(int))
        segmentation = data[:,-1]
        data = data[:,:-1]

        clusts = self.get_clusts(data, bids, segmentation, track_points)
        return clusts
