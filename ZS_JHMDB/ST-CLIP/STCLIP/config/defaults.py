from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()

_C.MODEL.WEIGHT = ""

# -----------------------------------------------------------------------------
# ROI action head config.
# -----------------------------------------------------------------------------
_C.MODEL.ROI_ACTION_HEAD = CN()

# Feature extractor config.
_C.MODEL.ROI_ACTION_HEAD.FEATURE_EXTRACTOR = "MaxpoolFeatureExtractor"
_C.MODEL.ROI_ACTION_HEAD.MLP_HEAD_DIM = 512

# Action predictor config.
_C.MODEL.ROI_ACTION_HEAD.PREDICTOR = "FCPredictor"
_C.MODEL.ROI_ACTION_HEAD.DROPOUT_RATE = 0.0

# Text feature generator config.
_C.MODEL.ROI_ACTION_HEAD.TEXT_FEATURE_GENERATOR = "CLIPencoder"
_C.MODEL.ROI_ACTION_HEAD.PREFIX_LEN = 16
_C.MODEL.ROI_ACTION_HEAD.POSTFIX_LEN = 16

# Video prompt
_C.MODEL.ROI_ACTION_HEAD.VIDEO_PROMPT = "VideoSpecificPrompt"
_C.MODEL.ROI_ACTION_HEAD.VIDEO_FEATURE_GENERATOR = "MIT"

# Action loss evaluator config.
_C.MODEL.ROI_ACTION_HEAD.PROPOSAL_PER_CLIP = 10
# _C.MODEL.ROI_ACTION_HEAD.NUM_PERSON_MOVEMENT_CLASSES = 21
_C.MODEL.ROI_ACTION_HEAD.NUM_PERSON_MOVEMENT_CLASSES = 15 # zero shot: 15
_C.MODEL.ROI_ACTION_HEAD.NUM_OBJECT_MANIPULATION_CLASSES = 0
_C.MODEL.ROI_ACTION_HEAD.NUM_PERSON_INTERACTION_CLASSES = 0

_C.MODEL.ROI_ACTION_HEAD.POSE_LOSS_WEIGHT = 1.2
_C.MODEL.ROI_ACTION_HEAD.OBJECT_LOSS_WEIGHT = float(_C.MODEL.ROI_ACTION_HEAD.NUM_OBJECT_MANIPULATION_CLASSES)
_C.MODEL.ROI_ACTION_HEAD.PERSON_LOSS_WEIGHT = float(_C.MODEL.ROI_ACTION_HEAD.NUM_PERSON_INTERACTION_CLASSES)
_C.MODEL.ROI_ACTION_HEAD.POSE_LOSS_WEIGHT = 1.0

# Focal loss config.
_C.MODEL.ROI_ACTION_HEAD.FOCAL_LOSS = CN()
_C.MODEL.ROI_ACTION_HEAD.FOCAL_LOSS.GAMMA = 2.0
_C.MODEL.ROI_ACTION_HEAD.FOCAL_LOSS.ALPHA = -1.

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = 256
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 464
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 256
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 464
# Values to be used for image normalization, in rgb order
_C.INPUT.PIXEL_MEAN = [122.7717, 115.9465, 102.9801]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [57.375, 57.375, 57.375]
# Convert image to BGR format (for Caffe2 models)
_C.INPUT.TO_BGR = False

_C.INPUT.FRAME_NUM = 32
_C.INPUT.FRAME_SAMPLE_RATE = 2
_C.INPUT.TAU = 8
_C.INPUT.ALPHA = 8
_C.INPUT.SLOW_JITTER = False

_C.INPUT.COLOR_JITTER = False
_C.INPUT.HUE_JITTER = 20.0 #in degree, hue is in 0~360
_C.INPUT.SAT_JITTER = 0.1
_C.INPUT.VAL_JITTER = 0.1

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# TRAIN_LABEL
# -----------------------------------------------------------------------------
_C.ALL_LABEL = ""
_C.TRAIN_LABEL = ""
_C.TEST_LABEL = ""
_C.TRAIN_VIDEO = ""
_C.TEST_VIDEO = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of dataset loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 16
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True

_C.MODEL.STCLIP_structure = CN()
_C.MODEL.STCLIP_structure.ACTIVE = False
_C.MODEL.STCLIP_structure.STRUCTURE = "serial"
_C.MODEL.STCLIP_structure.MAX_PER_SEC = 5
_C.MODEL.STCLIP_structure.MAX_PERSON = 25
_C.MODEL.STCLIP_structure.MAX_KEYPOINTS = 25
_C.MODEL.STCLIP_structure.DIM_IN = 2304
_C.MODEL.STCLIP_structure.DIM_INNER = 512
_C.MODEL.STCLIP_structure.DIM_OUT = 512
_C.MODEL.STCLIP_structure.PENALTY = True
_C.MODEL.STCLIP_structure.LENGTH = (30, 30)
_C.MODEL.STCLIP_structure.MEMORY_RATE = 1
_C.MODEL.STCLIP_structure.CONV_INIT_STD = 0.01
_C.MODEL.STCLIP_structure.DROPOUT = 0.
_C.MODEL.STCLIP_structure.NO_BIAS = False
_C.MODEL.STCLIP_structure.I_BLOCK_LIST = ['P', 'O', 'H', 'M', 'P', 'O', 'H', 'M']
_C.MODEL.STCLIP_structure.LAYER_NORM = False
_C.MODEL.STCLIP_structure.TEMPORAL_POSITION = True
_C.MODEL.STCLIP_structure.ROI_DIM_REDUCE = True
_C.MODEL.STCLIP_structure.USE_ZERO_INIT_CONV = True
_C.MODEL.STCLIP_structure.MAX_OBJECT = 0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 75000

_C.SOLVER.BASE_LR = 0.02
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.IA_LR_FACTOR = 1.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

# Use for bn
_C.SOLVER.WEIGHT_DECAY_BN = 0.0

_C.SOLVER.SCHEDULER = "warmup_multi_step"

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (33750, 67500)

_C.SOLVER.WARMUP_ON = True
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 5000
_C.SOLVER.EVAL_PERIOD = 5000

# Number of video clips per batch
# This is global, so if we have 8 GPUs and VIDEOS_PER_BATCH = 16, each GPU will
# see 2 clips per batch
_C.SOLVER.VIDEOS_PER_BATCH = 32


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of video clips per batch
# This is global, so if we have 8 GPUs and VIDEOS_PER_BATCH = 16, each GPU will
# see 2 clips per batch
_C.TEST.VIDEOS_PER_BATCH = 16

# Config used in inference.
_C.TEST.EXTEND_SCALE = (0.1, 0.05)
_C.TEST.BOX_THRESH = 0.8
_C.TEST.ACTION_THRESH = 0.05

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
