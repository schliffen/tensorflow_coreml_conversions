from __future__ import print_function
from __future__ import division

import argparse
import os
import numpy as np
import tensorflow as tf
# import cv2
import copy as cp
import math
import sys

from keras import callbacks
from keras import models
from keras.models import Model
from keras.models import load_model
from keras.layers import Reshape
from keras.backend import one_hot
from keras.backend import cast
from keras import losses
from keras.optimizers import SGD, Adam
from keras.utils import multi_gpu_model
from scipy.misc import imresize
from scipy.ndimage import imread

from learning_rate import create_lr_schedule
from loss import dice_coef_loss, dice_coef, recall, precision, softmax_loss
from nets.MobileUnet_01 import MobileUNet_01

from keras.utils.training_utils import multi_gpu_model

checkpoint_path = './artifacts/checkpoint_weights.{epoch:02d}-{val_loss:.2f}.h5'
trained_model_path = './artifacts/model.h5'

nb_train_samples = 2664
nb_validation_samples = 666

# transform an onehot array into an array with a gaussian kernel
sigma = 1
gamma = 3


# get gaussian label
def gaussian(labels):
    numofdata = labels.shape[0]
    temp = np.zeros((numofdata, 5, 224 * 224))
    summation = np.zeros((numofdata, 1, 224 * 224))
    table = np.zeros(199809)  # 447*447
    table = calculate_table(table)

    for i in range(numofdata):
        print(i)
        for j in range(5):
            y = int(labels[i][j][1])
            x = int(labels[i][j][0])
            temp[i][j] = get_table(table, x, y)
            summation[i][0] = temp[i][j] + summation[i][0]
        s = np.sum(summation[i][0])
        print(str(s))
        summation[i][0] = summation[i][0] / s
    result = np.reshape(summation, (numofdata, 1, 224, 224))
    with open('label.txt', 'w') as f:
        for i in range(224):
            for j in range(224):
                f.write(str(result[0][0][i][j]) + '\n')
    print('label written!')
    return result

    # create gaussian table,only need to calculate 1/4 of it(can improve to 1/8)


def calculate_table(table):
    for i in range(224):
        for j in range(224):
            G = cauchy_distribution(i, j, 223, 223)
            table[i + j * 447] = G
            table[446 - i + j * 447] = G
            table[i + (446 - j) * 447] = G
            table[446 - i + (446 - j) * 447] = G
    # with open('Cauchytable.txt','w') as f:
    # for i in range(199809):
    # f.write(str(table[i])+'\n')
    # if ((i + 1) % 447) == 0:
    # f.write('\n')
    return table

    # x0,y0 is the original point. Calculate gaussian distribution.


def gaussian_distribution(x, y, x0, y0):
    G = math.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 2)
    # print(G)
    return G


def cauchy_distribution(x, y, x0, y0):
    C = (1 / (2 * math.pi)) * (gamma / (((x - x0) ** 2 + (y - y0) ** 2 + gamma ** 2) ** 1.5))
    return C

    # slice a 224*224 table from the 447*447 gaussian table


def get_table(table, x, y):
    part = np.zeros(50176)
    for j in range(224):
        part[(j * 224):(j * 224 + 223)] = table[(223 - x + 447 * (223 - y + j)):(446 - x + 447 * (223 - y + j))]
    return part

    # data and label are generated here
    # output is (1,224,224,3) and (1,1) array


def generator(data, label, batch_size):
    Numofbatch = data.shape[0] // batch_size
    tempindex = np.arange(0, data.shape[0])
    np.random.shuffle(tempindex)
    cpdata = cp.deepcopy(data)
    cplabel = cp.deepcopy(label)
    for j in range(data.shape[0]):
        cpdata[tempindex[j]] = data[j]
        cplabel[tempindex[j]] = label[j]
    while 1:
        for i in range(Numofbatch):
            yield cpdata[i * batch_size:(i + 1) * batch_size], cplabel[i * batch_size:(i + 1) * batch_size]


def train(mobilenet_weights_path, epochs, batch_size):
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)tensorflow export model for 	    inference with batch size 1

    # file containing the path to images and labels
    filename = './dataset.txt'
    filenames = []
    labels1 = []
    labels2 = []
    labels3 = []
    labels4 = []
    labels5 = []
    # reading file and extracting path and labels
    with open(filename, 'r') as File:
        infoFile = File.readlines()  # reading lines from files
        for line in infoFile:  # reading line by line
            words = line.split(' ')
            filenames.append(words[0])
            tempy1 = int(int(words[2]))
            tempx1 = int(int(words[1]))
            tempy2 = int(int(words[4]))
            tempx2 = int(int(words[3]))
            tempy3 = int(int(words[6]))
            tempx3 = int(int(words[5]))
            tempy4 = int(int(words[8]))
            tempx4 = int(int(words[7]))
            tempy5 = int(int(words[10]))
            tempx5 = int(int(words[9]))
            label1 = []
            label2 = []
            label3 = []
            label4 = []
            label5 = []
            # print(words[0])
            label1.append(tempx1)
            label1.append(tempy1)
            labels1.append(label1)
            label2.append(tempx2)
            label2.append(tempy2)
            labels2.append(label2)
            label3.append(tempx3)
            label3.append(tempy3)
            labels3.append(label3)
            label4.append(tempx4)
            label4.append(tempy4)
            labels4.append(label4)
            label5.append(tempx5)
            label5.append(tempy5)
            labels5.append(label5)
            # labels.append(int(words[1]))
            # labels.append(int(words[2]))
            # labels.append(int(words[3]))
            # labels.append(int(words[4]))
            # labels.append(int(words[5]))
            # labels.append(int(words[6]))
            # labels.append(int(words[7]))
            # labels.append(int(words[8]))
            # labels.append(int(words[9]))
            # labels.append(int(words[10]))

    input_image = []
    NumFiles = len(filenames)
    lbl1 = []
    lbl2 = []
    lbl3 = []
    lbl4 = []
    lbl5 = []

    lbl_train_array = []
    img_train_array = []

    lbl_val_array = []
    img_val_array = []
    #  create lists contain data and labels
    # 0.8 train data, 0.2 validation data
    print(NumFiles)
    i = 0
    # labels1 =np.reshape(labels1, (-1,1))
    # labels2 =np.reshape(labels2, (-1,1))
    # labels3 =np.reshape(labels3, (-1,1))
    # labels4 =np.reshape(labels4, (-1,1))
    # labels5 =np.reshape(labels5, (-1,1))
    for i in range(NumFiles):
        lbl = []
        lbl1 = np.cast['f'](labels1[i])
        lbl2 = np.cast['f'](labels2[i])
        lbl3 = np.cast['f'](labels3[i])
        lbl4 = np.cast['f'](labels4[i])
        lbl5 = np.cast['f'](labels5[i])
        lbl.append(lbl1)
        lbl.append(lbl2)
        lbl.append(lbl3)
        lbl.append(lbl4)
        lbl.append(lbl5)
        imageO = imread(filenames[i])
        # print(filenames[i])
        imageO = np.cast['f'](imageO)
        imageO = imresize(imageO, (224, 224))
        # input_image = np.reshape(imageO, (150528))
        if i < NumFiles * 0.8:
            lbl_train_array.append(lbl)
            img_train_array.append(imageO)
        else:
            lbl_val_array.append(lbl)
            img_val_array.append(imageO)
    # convert list to array
    # convert label array to onehot array

    train_data = (np.asarray(img_train_array))
    train_labels = (np.asarray(lbl_train_array))
    sess = tf.InteractiveSession()
    train_gaussian_labels = gaussian(train_labels)
    train_generator = generator(train_data, train_gaussian_labels, batch_size)
    # train_generator = generator(train_data, train_onehot_labels, batch_size)

    val_data = (np.asarray(img_val_array))
    val_labels = (np.asarray(lbl_val_array))
    val_gaussian_labels = gaussian(val_labels)
    val_generator = generator(val_data, val_gaussian_labels, batch_size)
    # val_generator = generator(val_data, val_onehot_labels, batch_size)
    # train_data = np.transpose(np.asarray(img_array))
    # train_labels = np.transpose(np.asarray(lbl_array))
    print(train_data.shape)
    print(train_gaussian_labels.shape)

    lr_base = 0.01 * (float(batch_size) / 16)

    # model is bulit here

    prev_model = MobileUNet_01(input_shape=(224, 224,3), alpha_up=1)
    prev_model.summary()
    top_model = models.Sequential()
    top_model.add(Reshape((1, 224, 224), input_shape=(None, 224, 224, 3), name='softmax_tensor'))
    top_model.summary()
    model = Model(inputs=prev_model.input, outputs=top_model(prev_model.output))

    multi_model = multi_gpu_model(prev_model, gpus=2)




    # model = MobileDeepLab(input_shape=(img_height, img_width, 3))
    multi_model.load_weights(os.path.expanduser(mobilenet_weights_path.format(224)), by_name=True)

    # Freeze above conv_dw_12
    for layer in multi_model.layers[:70]:
        layer.trainable = True

    # Freeze above conv_dw_13
    # for layer in model.layers[:76]:
    #     layer.trainable = False

    multi_model.summary()
    # model_multi = multi_gpu_model(model, gpus=4)
    multi_model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        # optimizer=Adam(lr=0.0001),
        loss=softmax_loss,
        metrics=[
            recall,
            precision,
            'accuracy',
            'mean_squared_error',
        ])

    # callbacks
    scheduler = callbacks.LearningRateScheduler(
        create_lr_schedule(epochs, lr_base=lr_base, mode='progressive_drops'))
    tensorboard = callbacks.TensorBoard(log_dir='./logs')
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=True,
                                           save_best_only=True)

    multi_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[scheduler, tensorboard, checkpoint],
    )

    multi_model.save(trained_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mobilenet_weights_path',
        type=str,
        default='./artifacts/mobilenet_1_0_224_tf.h5',
        help='mobilenet weights using imagenet which is available at keras page'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
    )
    args, _ = parser.parse_known_args()

    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    train(**vars(args))
