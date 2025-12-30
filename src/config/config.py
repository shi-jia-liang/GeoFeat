from yacs.config import CfgNode as CN

## Configuration for data paths and settings
cfg = CN()

##############  ↓  DATA CONFIG  ↓  ##############
cfg.DATASET = CN()
cfg.DATASET.NUM_WORKERS = 10
cfg.DATASET.USE_MEGADEPTH = True
cfg.DATASET.USE_COCO = True
cfg.DATASET.MEGADEPTH_ROOT_PATH = "D:/DataSets/MegaDepth"
cfg.DATASET.MEGADEPTH_BATCH_SIZE = 2
cfg.DATASET.COCO_ROOT_PATH = "./datasets/coco_20k"
cfg.DATASET.COCO_BATCH_SIZE = 2

##############  ↓  MODEL CONFIG  ↓  ##############
cfg.MODEL = CN()

# Backbone options: "Standard", "RepVGG"
cfg.MODEL.BACKBONE = "Standard"        

# Upsample options: "bilinear", "pixel_shuffle"
# Bilinear Upsampling (双线性插值)
# PixelShuffle (亚像素卷积)
cfg.MODEL.UPSAMPLE_TYPE = "bilinear"
# Positional Encoding options: "None", "fourier","rot_inv"
cfg.MODEL.POS_ENC_TYPE = "None"

cfg.MODEL.KEYPOINT_ENCODER = [128, 64, 64]
cfg.MODEL.KEYPOINT_DIM = 65

cfg.MODEL.DESCRIPTOR_ENCODER = [64, 64]
cfg.MODEL.DESCRIPTOR_DIM = 64

# Normal, Gradient, and Curvature feature encoders
cfg.MODEL.GEOMETRIC_FEATURES = CN()
cfg.MODEL.GEOMETRIC_FEATURES.DEPTH = False
cfg.MODEL.GEOMETRIC_FEATURES.NORMAL = True
cfg.MODEL.GEOMETRIC_FEATURES.GRADIENTS = False
cfg.MODEL.GEOMETRIC_FEATURES.CURVATURES = False

cfg.MODEL.DEPTH_ENCODER = [128, 64, 64]


cfg.MODEL.NORMAL_ENCODER = [128, 64, 64]
cfg.MODEL.NORMAL_DIM = 192

cfg.MODEL.GRADIENT_ENCODER = [64, 32, 32]
cfg.MODEL.GRADIENT_DIM = 96

cfg.MODEL.CURVATURE_ENCODER = [64, 32, 32]
cfg.MODEL.CURVATURE_DIM = 96

# Swin-specific
cfg.MODEL.SWIN = CN()
cfg.MODEL.SWIN.INPUT_RESOLUTION = [75, 75]
cfg.MODEL.SWIN.DEPTH_PER_LAYER = 2
cfg.MODEL.SWIN.NUM_HEADS = 8
cfg.MODEL.SWIN.WINDOW_SIZE = 5

# Attention configuration
# ATTENTION.TYPE options: "AFT", "Swin"
cfg.MODEL.ATTENTION = CN()
cfg.MODEL.ATTENTIONAL_LAYERS = 3
cfg.MODEL.ATTENTION.TYPE = "AFT"     # "AFT" 或 "Swin"
cfg.MODEL.ATTENTION.AFT.FFN_TYPE = "positionwiseFFN" 	# 当 TYPE="AFT" 时可选 "positionwiseFFN"(PPN) 或 "Swin"
cfg.MODEL.ATTENTION.SWIN.FFN_TYPE = "SwinFFN" 			# 当 TYPE="Swin" 时强制 "Swin"


cfg.MODEL.LAST_ACTIVATION = "sigmoid"
cfg.MODEL.L2_NORMALIZATION = False
cfg.MODEL.USE_COORD_LOSS = False

cfg.MODEL.OUTPUT_DIM = 64

def get_cfg_defaults():
	"""Get a yacs CfgNode object with default values for my_project."""
	return cfg.clone()