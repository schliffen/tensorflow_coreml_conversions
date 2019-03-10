#
#
#
import sys, os
import numpy as np
import tensorflow as tf

from os.path import dirname
from tensorflow.core.framework import graph_pb2


import label_pb2
from google.protobuf import text_format

label_def = label_pb2.LabelDef()
ml_model_path = '/home/rapsodo/CLionProjects/tensorflow-coreML/out_coreml_model/'

with open('annotations/labels/ball_label_map.pbtxt', 'rb') as f:
    text_format.Merge(f.read(), label_def)

with open('labels.txt', 'wb') as f:
    f.write('???\n')
    for item in label_def.item:
        f.write(item.name + '\n')


# Load the TF graph definition
#tf_model_path = '/home/rapsodo/CLionProjects/rapsodo_ball_detection/models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_SSDMOBNETV2.pb'
#with open(tf_model_path, 'rb') as f:
#    serialized = f.read()
#tf.reset_default_graph()
#original_gdef = tf.GraphDef()
#original_gdef.ParseFromString(serialized)

#with tf.Graph().as_default() as g:
#    tf.import_graph_def(original_gdef, name='')


# Strip unused subgraphs and save it as another frozen TF model
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
# input_node_names = ['Preprocessor/sub']
# output_node_names = ['concat', 'concat_1']
#gdef = strip_unused_lib.strip_unused(
#        input_graph_def = original_gdef,
#        input_node_names = input_node_names,
#        output_node_names = output_node_names,
#        placeholder_type_enum = dtypes.float32.as_datatype_enum)
# Save the feature extractor to an output file
# frozen_model_file = 'core_ml_conversion/ssd_mobilenet_feature_extractor.pb'
frozen_model_file = '/home/rapsodo/CLionProjects/rapsodo_ball_detection/FrozenGraph/ball_detect_02.pb'
#with gfile.GFile(frozen_model_file, "wb") as f:
#    f.write(gdef.SerializeToString())


# Now we have a TF model ready to be converted to CoreML
import tfcoreml
# Supply a dictionary of input tensors' name and shape (with # batch axis)
input_tensor_shapes = {"image_tensor:0":[1,130,130,3]} # batch size is 1
# Output CoreML model path
coreml_model_file = ml_model_path + 'ssd_mobilenet_03.mlmodel'
# The TF model's ouput tensor name
output_tensor_names = ['detection_scores:0', 'detection_boxes:0']

coreml_model = tfcoreml.convert(
      tf_model_path=frozen_model_file,
      mlmodel_path=coreml_model_file,
      input_name_shape_dict=input_tensor_shapes,
      image_input_names="image_tensor:0",
      output_feature_names=output_tensor_names,
      image_scale=1./255.,
      red_bias=-1.0,
      green_bias=-1.0,
      blue_bias=-1.0
)

os.remove(frozen_model_file)
