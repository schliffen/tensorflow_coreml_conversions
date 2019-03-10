# Copyright 2018 Zihua Zeng (edvard_hua@live.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
from networks import get_network
import os

from pprint import pprint

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
parser.add_argument('--model', type=str, default='mv2_hourglass', help='mv2_cpm or ')
parser.add_argument('--size', type=int, default=[192, 192])
parser.add_argument('--checkpoint', type=str,
                    default='/home/rapsodo/CLionProjects/point_detection/training/src/training/hrglss/model/1/model-3400',
                    help='checkpoint path')
parser.add_argument('--output_node_names', type=str, default='hourglass_out_4'
                    #"output layer for model mv2_cmp: Convolutional_Pose_Machine_1/stage_5_out:0"
                    # Convolutional_Pose_Machine/stage_0_out
                    )
parser.add_argument('--output_graph', type=str, default='/home/rapsodo/CLionProjects/point_detection/training/exports/hrglss/testing_hrglss_model_01.pb', help='output_freeze_path')

args = parser.parse_args()

input_node = tf.placeholder(tf.float32, shape=[1, args.size[1], args.size[0], 3], name="image")

# with tf.Session() as sess:
#     net = get_network(args.model, input_node, trainable=False)
#     saver = tf.train.Saver()
#     saver.restore(sess, args.checkpoint)
#
#     input_graph_def = tf.get_default_graph().as_graph_def()
#     output_graph_def = tf.graph_util.convert_variables_to_constants(
#         sess,  # The session
#         input_graph_def,  # input_graph_def is useful for retrieving the nodes
#         args.output_node_names.split(",")
#     )
#     saver = tf.train.import_meta_graph(args.checkpoint + '.meta', clear_devices=False)
#
#

with tf.Session(graph=tf.Graph()) as sess:
    net = get_network(args.model, input_node, trainable=False)
    saver = tf.train.saver()
    saver.restore(sess, args.checkpoint)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        args.output_node_names.split(',') # unrelated nodes will be discarded
    )
    #

    with tf.gfile.GFile(args.output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

print("%d ops in the final graph." % len(output_graph_def.node))




