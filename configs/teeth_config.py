from yacs.config import CfgNode as CN

'''
Configuration
'''
_T = CN()

_T.MODEL = CN()
_T.NUM_CLASSES = 2
_T.THRES_RADIUS = [5.0, 15.0]
_T.RADIUS_MEDIAN = 7.0
_T.ROUND_FACTOR = 100
_T.LAMBDA_C = 30
_T.CHECKPOINT_PATH = './experiments'
_T.DECIMATION_FACTOR = 50000
_T.MODEL_CONFIG = [{
    1:3,
    2:3,
    3:3,
    4:2,
    5:2,
    6:1,
    7:1,
    8:1,
    9:1,
    10:1,
    11:1,
    12:2,
    13:2,
    14:3,
    15:3,
    16:3
}]
_T.LOWER_LABEL = [{
    0:   0,
    1:  38,
    2:  37,
    3:  36,
    4:  35,
    5:  34,
    6:  33,
    7:  32,
    8:  31,
    9:  41,
    10: 42,
    11: 43,
    12: 44,
    13: 45,
    14: 46,
    15: 47,
    16: 48
}]
_T.UPPER_LABEL = [{
    0:   0,
    1:  18,
    2:  17,
    3:  16,
    4:  15,
    5:  14,
    6:  13,
    7:  12,
    8:  11,
    9:  21,
    10: 22,
    11: 23,
    12: 24,
    13: 25,
    14: 26,
    15: 27,
    16: 28
}]