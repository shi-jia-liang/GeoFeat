from yacs.config import CfgNode as CN

## Configuration for data paths and settings
cfg = CN()

##############  ↓  DATA CONFIG  ↓  ##############
cfg.DATASET = CN()
cfg.DATASET.NUM_WORKERS = 0
cfg.DATASET.USE_MEGADEPTH = True
cfg.DATASET.USE_COCO = True
cfg.DATASET.MEGADEPTH_ROOT_PATH = "./datasets/MegaDepth"
cfg.DATASET.MEGADEPTH_BATCH_SIZE = 2
cfg.DATASET.COCO_ROOT_PATH = "./datasets/coco_20k"
cfg.DATASET.COCO_BATCH_SIZE = 2

##############  ↓  MODEL CONFIG  ↓  ##############
cfg.MODEL = CN()

# Backbone options: "Standard", "RepVGG"
cfg.MODEL.BACKBONE = "RepVGG"        

# Upsample options: "bilinear", "pixel_shuffle"
# Bilinear Upsampling (双线性插值)
# PixelShuffle (亚像素卷积)
cfg.MODEL.UPSAMPLE_TYPE = "bilinear"

cfg.MODEL.KEYPOINT_DIM = 65
cfg.MODEL.KEYPOINT_ENCODER = [128, 64, 64]

cfg.MODEL.DESCRIPTOR_DIM = 64
cfg.MODEL.DESCRIPTOR_ENCODER = [64, 64]

# Normal, Gradient, and Curvature feature encoders
cfg.MODEL.GEOMETRIC_FEATURES = CN()
cfg.MODEL.GEOMETRIC_FEATURES.DEPTH = True
cfg.MODEL.GEOMETRIC_FEATURES.NORMAL = True
cfg.MODEL.GEOMETRIC_FEATURES.GRADIENTS = True
cfg.MODEL.GEOMETRIC_FEATURES.CURVATURES = True

cfg.MODEL.DEPTH_ENCODER = [128, 64, 64]
cfg.MODEL.DEPTH_DIM = 64


cfg.MODEL.NORMAL_ENCODER = [128, 64, 64]
cfg.MODEL.NORMAL_DIM = 192

cfg.MODEL.GRADIENT_ENCODER = [64, 32, 32]
cfg.MODEL.GRADIENT_DIM = 128

cfg.MODEL.CURVATURE_ENCODER = [64, 32, 32]
cfg.MODEL.CURVATURE_DIM = 320

# cfg.MODEL.DEPTH_ENCODER = [128, 64, 64]
# cfg.MODEL.DEPTH_DIM = 64


# cfg.MODEL.NORMAL_ENCODER = [128, 64, 64]
# cfg.MODEL.NORMAL_DIM = 192

# cfg.MODEL.GRADIENT_ENCODER = [64, 32, 32]
# cfg.MODEL.GRADIENT_DIM = 64

# cfg.MODEL.CURVATURE_ENCODER = [64, 32, 32]
# cfg.MODEL.CURVATURE_DIM = 160

# Attention configuration
cfg.MODEL.ATTENTIONAL_LAYERS = 3
cfg.MODEL.ATTENTION_FFN_TYPE = "positionwiseFFN"

# Refiner options: "None", "Local", "Geometric"
# "Local" uses simple depthwise separable conv
# "Geometric" uses geometry-guided refinement (Requires geometric features)
cfg.MODEL.REFINER_TYPE = "None"

# Final activation for descriptors: "None", "relu", "sigmoid", "tanh"
cfg.MODEL.LAST_ACTIVATION = "sigmoid"

cfg.MODEL.OUTPUT_DIM = 64

def get_cfg_defaults():
	"""Get a yacs CfgNode object with default values for my_project."""
	return cfg.clone()
