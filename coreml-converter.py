import argparse
import re

import coremltools
from keras.models import load_model
from keras.utils import CustomObjectScope
import keras

from coreml.hack import hack_coremltools
from MobileUNet import custom_objects
from learning_rate import create_lr_schedule
from loss import dice_coef_loss, dice_coef, recall, precision, softmax_loss
from MobileUNet import MobileUNet


def main(input_model_path):
    """
    Convert hdf5 file to CoreML model.
    :param input_model_path:
    :return:
    """
    out_path = re.sub(r"h5$", 'mlmodel', input_model_path)

    hack_coremltools()
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D':
keras.applications.mobilenet.DepthwiseConv2D,'softmax_loss':softmax_loss,'recall':
recall,'precision':precision},custom_objects()):
        model = load_model(input_model_path)
        coreml_model = coremltools.converters.keras.convert(model)
    coreml_model.save(out_path)

    print('CoreML model is created at %s' % out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_model_path',
        type=str,
        default='./artifacts/model.h5',
    )
    args, _ = parser.parse_known_args()

    main(**vars(args))
