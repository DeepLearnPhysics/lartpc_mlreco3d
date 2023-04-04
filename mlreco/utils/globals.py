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

# Convention for particle type labels 
PARTICLE_TO_PID_LABEL = {
    'PHOTON': 0,
    'ELECTRON': 1,
    'MUON': 2,
    'PION': 3,
    'PROTON': 4
}

PID_LABEL_TO_PARTICLE = {val : key for key, val in PARTICLE_TO_PID_LABEL.items()}

# CONSTANTS (MeV)
PROTON_MASS = 938.272
MUON_MASS = 105.7
ELECTRON_MASS = 0.511998
ARGON_DENSITY = 1.396
ADC_TO_MEV = 1. / 350.
PIXELS_TO_CM = 0.3

# Shape ID of each type of voxel category
SHOW_SHP   = larcv.kShapeShower    # 0
TRACK_SHP  = larcv.kShapeTrack     # 1
MICH_SHP   = larcv.kShapeMichel    # 2
DELTA_SHP  = larcv.kShapeDelta     # 3
LOWE_SHP   = larcv.kShapeLEScatter # 4
GHOST_SHP  = larcv.kShapeGhost     # 5
UNKWN_SHP  = larcv.kShapeUnknown   # 6

# Shape precedence used in the cluster labeling process
SHAPE_PREC = [TRACK_SHP, MICH_SHP, SHOW_SHP, DELTA_SHP, LOWE_SHP]

# Invalid labels
INVAL_TID  = larcv.kINVALID_UINT

# Mapping between particle PDG code and particle ID labels
PDG_TO_PID = {
    22:   0,  # photon
    11:   1,  # e-
    -11:  1,  # e+
    13:   2,  # mu-
    -13:  2,  # mu+
    211:  3,  # pi+
    -211: 3,  # pi-
    2212: 4,  # protons
}
