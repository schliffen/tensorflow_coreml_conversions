# -*- coding: utf-8 -*-
#
# Nodified by Ali
#

import argparse
import os


os.environ['CUDA_VISIBLE_DEVICES'] = ''
parser = argparse.ArgumentParser(description="Tools for convert frozen_pb into tflite or coreml.")
parser.add_argument("--frozen_pb", type=str, default='/home/rapsodo/CLionProjects/point_detection/training/src/training/model/testing_model_05.pb', help="Path for storing frozen graph.")
parser.add_argument("--input_node_name", type=str, default="IteratorGetNext", help="Name of input node name.")
parser.add_argument("--output_node_name", type=str, default="Convolutional_Pose_Machine/stage_5_out", help="Name of output node name.")
parser.add_argument("--output_path", type=str, default="/home/rapsodo/CLionProjects/point_detection/training/src/training/model/", help="Path for storing tflite & coreml")
parser.add_argument("--type", type=str, default="tflite", help="tflite or coreml")

args = parser.parse_args()

output_filename = args.frozen_pb.rsplit("/", 1)[1]
# output_filename = args.frozen_pb.rsplit("/", 1)[1]
output_filename = output_filename.split(".")[0]


if "tflite" in args.type:
    import tensorflow as tf
    from tensorflow.contrib import lite
    output_filename += ".tflite"
    converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
        args.frozen_pb,
        [args.input_node_name],
        [args.output_node_name]
    )
    # converter = lite.TFLiteConverter.from_frozen_graph(
    #     args.frozen_pb,
    #     [args.input_node_name],
    #     [args.output_node_name]
    # )

    tflite_model = converter.convert()
    open(os.path.join(args.output_path, output_filename), "wb").write(tflite_model)
    print("Generate tflite success.")
elif "coreml" in args.type:
    import tfcoreml as tf_converter
    output_filename += ".mlmodel"
    tf_converter.convert(tf_model_path=args.frozen_pb,
                         mlmodel_path = os.path.join(args.output_path, output_filename),
                         image_input_names = ["%s:0" % args.input_node_name],
                         output_feature_names = ['%s:0' % args.output_node_name])
    print("Generate CoreML success.")





