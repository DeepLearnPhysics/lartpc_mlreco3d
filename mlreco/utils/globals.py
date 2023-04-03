from larcv import larcv

# Column which specifies the batch ID in a sparse tensor
BATCH_COL  = 0

# Columns which specify the voxel coordinates in a sparse tensor
COORD_COLS = (1,2,3)

# Colum which specifies the first value of a voxel in a sparse tensor
VALUE_COL  = 4

# Columns that specify each attribute in a cluster label tensor
CLUST_COL  = 5
GROUP_COL  = 6
INTER_COL  = 7
NU_COL     = 8
TYPE_COL   = 9
PSHOW_COL  = 10
PGRP_COL   = 11
VTX_COLS   = (12,13,14)
MOM_COL    = 15

# Colum which specifies the shape ID of a voxel in a sparse tensor 
SHAPE_COL  = -1

# Shape ID of each type of voxel category
SHOW_SHP   = larcv.kShapeShower    # 0
TRACK_SHP  = larcv.kShapeTrack     # 1
MICH_SHP   = larcv.kShapeMichel    # 2
DELTA_SHP  = larcv.kShapeDelta     # 3
LOWE_SHP   = larcv.kShapeLEScatter # 4
GHOST_SHP  = larcv.kShapeGhost     # 5
UNKWN_SHP  = larcv.kShapeUnknown   # 6

# Shape precedence in cluster labels
SHAPE_PREC = [TRACK_SHP, MICH_SHP, SHOW_SHP, DELTA_SHP, LOWE_SHP]
