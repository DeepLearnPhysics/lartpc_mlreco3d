import numpy as np
from collections import defaultdict
from larcv import larcv

# Column which specifies the batch ID in a sparse tensor
BATCH_COL = 0

# Columns which specify the voxel coordinates in a sparse tensor
COORD_COLS = np.array([1, 2, 3])

# Colum which specifies the first value of a voxel in a sparse tensor
VALUE_COL = 4

# Columns that specify each attribute in a cluster label tensor
CLUST_COL = 5
GROUP_COL = 6
INTER_COL = 7
NU_COL    = 8
PID_COL   = 9
PSHOW_COL = 10
PGRP_COL  = 11
VTX_COLS  = np.array([12, 13, 14])
MOM_COL   = 15
PART_COL  = 16 # TODO: change order

# Colum which specifies the shape ID of a voxel in a label tensor
SHAPE_COL = -1

# Columns that specify each output in a PPN output tensor
PPN_ROFF_COLS  = np.array([0, 1, 2])         # Raw offset
PPN_RTYPE_COLS = np.array([3, 4, 5, 6, 7])   # Raw class type scores
PPN_RPOS_COLS  = np.array([8, 9])            # Raw positive score

PPN_SCORE_COLS = np.array([4, 5])            # Softmax positive scores
PPN_OCC_COL    = 6                           # Occupancy score
PPN_CLASS_COLS = np.array([7, 8, 9, 10, 11]) # Softmax class scores
PPN_SHAPE_COL  = 12                          # Predicted shape
PPN_END_COLS   = np.array([13, 14])          # Softmax end point scores

# Shape ID of each type of voxel category
SHOWR_SHP = larcv.kShapeShower    # 0
TRACK_SHP = larcv.kShapeTrack     # 1
MICHL_SHP = larcv.kShapeMichel    # 2
DELTA_SHP = larcv.kShapeDelta     # 3
LOWES_SHP = larcv.kShapeLEScatter # 4
GHOST_SHP = larcv.kShapeGhost     # 5
UNKWN_SHP = larcv.kShapeUnknown   # 6

# Shape precedence used in the cluster labeling process
SHAPE_PREC = [TRACK_SHP, MICHL_SHP, SHOWR_SHP, DELTA_SHP, LOWES_SHP, UNKWN_SHP]

# Shape labels
SHAPE_LABELS = {
   0:  'Shower',
   1:  'Track',
   2:  'Michel',
   3:  'Delta',
   4:  'Low Energy',
   5:  'Ghost',
   6:  'Unknown'
}

# Invalid larcv.Particle labels
INVAL_ID   = larcv.kINVALID_INSTANCEID # Particle group/parent/interaction ID
INVAL_TID  = larcv.kINVALID_UINT       # Particle Geant4 track ID
INVAL_PDG  = 0                         # Particle PDG code

# Mapping between particle PDG code and particle ID labels
PDG_TO_PID = defaultdict(lambda: -1)
PDG_TO_PID.update({
    22:   0,  # photon
    11:   1,  # e-
    -11:  1,  # e+
    13:   2,  # mu-
    -13:  2,  # mu+
    211:  3,  # pi+
    -211: 3,  # pi-
    2212: 4,  # protons
    321:  5,  # K+
    -321: 5   # K-
})

PID_TO_PDG = {v : abs(k) for k, v in PDG_TO_PID.items()}

# Particle type labels
PID_LABELS = {
    0: 'Photon',
    1: 'Electron',
    2: 'Muon',
    3: 'Pion',
    4: 'Proton',
    5: 'Kaon'
}

# Map between shape and allowed PID/primary labels
SHP_TO_PID = {
    SHOWR_SHP: np.array([0,1]),
    TRACK_SHP: np.array([2,3,4,5]),
    DELTA_SHP: np.array([1]),
    MICHL_SHP: np.array([1])
}

SHP_TO_PRIMARY = {
    SHOWR_SHP: np.array([0,1]),
    TRACK_SHP: np.array([0,1]),
    DELTA_SHP: np.array([0]),
    MICHL_SHP: np.array([0])
}

# Particle masses
PID_MASSES = {
    0: 0.,
    1: 0.511998, # [MeV/c^2]
    2: 105.658,  # [MeV/c^2]
    3: 139.570,  # [MeV/c^2]
    4: 938.272,  # [MeV/c^2]
    5: 493.677   # [MeV/c^2]
}

# Neutrino current type
NU_CURR_TYPE = {
    -1: 'UnknownCurrent',
    0:  'CC',
    1:  'NC'
}

# Neutrino interaction mode and type labels
# Source: https://internal.dunescience.org/doxygen/MCNeutrino_8h_source.html
NU_INT_TYPE = {
    -1:   'UnknownInteraction',
    1:    'QE',
    2:    'DIS',
    3:    'Coh',
    4:    'CohElastic',
    5:    'ElectronScattering',
    6:    'IMDAnnihilation',
    7:    'InverseBetaDecay',
    8:    'GlashowResonance',
    9:    'AMNuGamma',
    10:   'MEC',
    11:   'Diffractive',
    12:   'EM',
    13:   'WeakMix',
    1000: 'NuanceOffset',
    1001: 'CCQE',
    1002: 'NCQE',
    1003: 'ResCCNuProtonPiPlus',
    1004: 'ResCCNuNeutronPi0',
    1005: 'ResCCNuNeutronPiPlus',
    1006: 'ResNCNuProtonPi0',
    1007: 'ResNCNuProtonPiPlus',
    1008: 'ResNCNuNeutronPi0',
    1009: 'ResNCNuNeutronPiMinus',
    1010: 'ResCCNuBarNeutronPiMinus',
    1011: 'ResCCNuBarProtonPi0',
    1012: 'ResCCNuBarProtonPiMinus',
    1013: 'ResNCNuBarProtonPi0',
    1014: 'ResNCNuBarProtonPiPlus',
    1015: 'ResNCNuBarNeutronPi0',
    1016: 'ResNCNuBarNeutronPiMinus',
    1017: 'ResCCNuDeltaPlusPiPlus',
    1021: 'ResCCNuDelta2PlusPiMinus',
    1028: 'ResCCNuBarDelta0PiMinus',
    1032: 'ResCCNuBarDeltaMinusPiPlus',
    1039: 'ResCCNuProtonRhoPlus',
    1041: 'ResCCNuNeutronRhoPlus',
    1046: 'ResCCNuBarNeutronRhoMinus',
    1048: 'ResCCNuBarNeutronRho0',
    1053: 'ResCCNuSigmaPlusKaonPlus',
    1055: 'ResCCNuSigmaPlusKaon0',
    1060: 'ResCCNuBarSigmaMinusKaon0',
    1062: 'ResCCNuBarSigma0Kaon0',
    1067: 'ResCCNuProtonEta',
    1070: 'ResCCNuBarNeutronEta',
    1073: 'ResCCNuKaonPlusLambda0',
    1076: 'ResCCNuBarKaon0Lambda0',
    1079: 'ResCCNuProtonPiPlusPiMinus',
    1080: 'ResCCNuProtonPi0Pi0',
    1085: 'ResCCNuBarNeutronPiPlusPiMinus',
    1086: 'ResCCNuBarNeutronPi0Pi0',
    1090: 'ResCCNuBarProtonPi0Pi0',
    1091: 'CCDIS',
    1092: 'NCDIS',
    1093: 'UnUsed1',
    1094: 'UnUsed2',
    1095: 'CCQEHyperon',
    1096: 'NCCOH',
    1097: 'CCCOH',
    1098: 'NuElectronElastic',
    1099: 'InverseMuDecay',
    1100: 'MEC2p2h'
}

# Physical constants
ARGON_DENSITY = 1.396     # [g/cm^3]
