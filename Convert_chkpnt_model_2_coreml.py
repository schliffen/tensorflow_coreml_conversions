#
# converting_feature extraction to coreml model
#
#
# By this code The feature extraction part is converting to coreml
#
import urllib, os, sys, zipfile
os.system('nvcc --version')
from os.path import dirname
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
import tfcoreml
import argparse
args = argparse.ArgumentParser()
args.add_argument('-d0', '--ckpnt', default='/home/rapsodo/workspace_mike3352/5p/last_checkpoint/', help='path to checkpoints')
args.add_argument('-d1', '--fg_dir', default= '/home/rapsodo/workspace_mike3352/5p/models_ex/', help='path to trained checkpoints')
args.add_argument('-s2', '--pb_dir', default= '/home/rapsodo/CLionProjects/rapsodo_ball_detection/FrozenGraph/mlmodels/FT_five_point_extractor_2.pb', help='path to frozen graph')
args.add_argument('-s3', '--cml_dir', default= '/home/ali/CLionProjects/rapsodo_ball_detection/FrozenGraph/mlmodels/ssd_mobilenet_feature_extractor_v2.mlmodel', help='path to frozen graph')
# args.add_argument('-s4', '--mdl_nm', default= 'model.ckpt-20000', help='checkpoint name if needed')

arg = args.parse_args()

#
# reading the check points
#
output_nodes = "tower_0/Softmax"
# checkpoint_file = tf.train.latest_checkpoint(arg.ckpnt + '2018-11-18 00_37_53_lr=0.001_lambda=0.001_bs=14-51')

input_checkpoint = arg.ckpnt + '2018-11-21 16:46:09_lr=0.001_lambda=0.001_bs=5.ckpnt-1-1'
# input_checkpoint = '/home/rapsodo/CLionProjects/rapsodo_ball_detection/ckpt/model.ckpt-72288'




#
dir(tf.contrib)

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(graph=tf.Graph(), config = config) as sess:
    with tf.device('/device:GPU:0'):
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=False)
        graph = tf.get_default_graph()
        onames = output_nodes.split(',')
        input_graph_def = graph.as_graph_def()
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
                            sess, input_graph_def, onames # unrelated nodes will be discarded
                            )
    #
    print('the graph is red and output node is set!')
    #
    with tf.gfile.GFile(arg.pb_dir, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final frozen graph." % len(output_graph_def.node))


with open(arg.pb_dir, 'rb') as f:
    serialized = f.read()
tf.reset_default_graph()
original_gdef = tf.GraphDef()
original_gdef.ParseFromString(serialized)

with tf.Graph().as_default() as g:
    tf.import_graph_def(original_gdef, name='')

# Strip unused subgraphs and save it as another frozen TF model
input_node_names = ['input_full']
output_node_names = ['tower_0/Softmax']
# gdef = strip_unused_lib.strip_unused(
#    input_graph_def = original_gdef,
#    input_node_names = input_node_names,
#    output_node_names = output_node_names,
#    placeholder_type_enum = dtypes.float32.as_datatype_enum)
# Save the feature extractor to an output file

# Supply a dictionary of input tensors' name and shape (with # batch axis)
input_tensor_shapes = {"input_full:0":[1,480,720,3]} # batch size is 1
#input_tensor_shapes = ['Preprocessor/sub:0'] # batch size is 1
# Output CoreML model path

# We retrieve the protobuf graph definition
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

coreml_model = tfcoreml.convert(
    tf_model_path=arg.pb_dir,
    mlmodel_path=arg.cml_dir,
    input_name_shape_dict=input_tensor_shapes,
    image_input_names="input_full:0",
    output_feature_names=output_node_names,
    image_scale=2./255.,
    red_bias=-1.0,
    green_bias=-1.0,
    blue_bias=-1.0
)
#
print('coreml file is created!')