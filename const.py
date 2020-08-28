"""
Constants for style transfer
"""
import os

from config import Config

TOTAL_VARIATION_WEIGHT = 30
EPSILON = 1e-1
BETA_1 = 0.99
BETA_2 = 0.999
LEARNING_RATE = 0.02
CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e-2
LOG_DIR = ".tf{}logs".format(os.path.sep)
DEFAULT_CONFIG = Config()