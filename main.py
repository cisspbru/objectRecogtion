#import necessary libraries
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from utils import label_maputil
from utils import visualiztion_utils as vis_util

#model file path
model_file = '/frozen_inference_graph.pb'
detection_graph = tf.Graph() 
with detection_graph.as_default():
  od_grapg_def = tf.GraphDef()
  with tf.gfile.GFile(model_file_name, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

    
 
label_map_file_name = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
label_map = label_map_util.load_labelmap(label_map_file_name)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)



with detection_graph.as_default():
  with tf.Session() as sess:
    ops = tf.get_default_graph().get_operations()
    all_tensor_name = {output.name for op in ops for output in op.outputes}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
     if tensor_name in all_tensor_names:
      detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
      detection_masks = tf.squeeze(tensor_dict['detection_masks], [0])
      real_num_detection_reframed = utils_ops_reframe_box_masks_to_image_masks(                                        
