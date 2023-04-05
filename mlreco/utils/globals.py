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
SHOWR_SHP  = larcv.kShapeShower    # 0
TRACK_SHP  = larcv.kShapeTrack     # 1
MICHL_SHP  = larcv.kShapeMichel    # 2
DELTA_SHP  = larcv.kShapeDelta     # 3
LOWES_SHP  = larcv.kShapeLEScatter # 4
GHOST_SHP  = larcv.kShapeGhost     # 5
UNKWN_SHP  = larcv.kShapeUnknown   # 6

# Shape precedence used in the cluster labeling process
SHAPE_PREC = [TRACK_SHP, MICHL_SHP, SHOWR_SHP, DELTA_SHP, LOWES_SHP]

# Invalid larcv.Particle labels
INVAL_ID   = larcv.kINVALID_INSTANCEID # Particle group/parent/interaction ID
INVAL_TID  = larcv.kINVALID_UINT       # Particle Geant4 track ID
INVAL_PDG  = 0                         # Patricle PDG code

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

# CONSTANTS
MUON_MASS     = 105.7     # [MeV/c^2]
ELECTRON_MASS = 0.511998  # [MeV/c^2]
PROTON_MASS   = 938.272   # [MeV/c^2]
ARGON_DENSITY = 1.396     # [g/cm^3]
ADC_TO_MEV    = 1. / 350. # < MUST GO
PIXELS_TO_CM  = 0.3       # < MUST GO
