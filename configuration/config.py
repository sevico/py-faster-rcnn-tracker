from easydict import EasyDict

__C = EasyDict()

cfg = __C

__C.TRAIN = EasyDict()

# Default Training Data Directory
__C.TRAIN.DATA_DIR = 'data'
# Snapshot Iterations
__C.TRAIN.SNAPSHOT_ITERS = 100000
# Batch Size
__C.TRAIN.IMS_PER_BATCH = 1
# Training Image Input Size
__C.TRAIN.INPUT_SIZE = 512

__C.TRAIN.USE_FLIPPED_IMAGE = True

__C.TRAIN.USE_TRANSPOSED_IMAGE = False

__C.TRAIN.SNAPSHOT_INFIX = ''

#####################################

__C.TEST = EasyDict()

# Defaut Testing Data Directory
__C.TEST.DATA_DIR = 'data'
# Testing Image Input Size
__C.TEST.INPUT_SIZE = 512
