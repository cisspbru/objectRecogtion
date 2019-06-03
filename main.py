import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from utils import label_maputil
from utils import visualiztion_utils as vis_util

#model file path
model_file = 'graph.pb'
detection_graph = tf.Graph()
