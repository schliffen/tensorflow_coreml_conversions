from __future__ import print_function
from __future__ import division

import argparse
import os
import numpy as np
import tensorflow as tf
#import cv2
import copy as cp

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
from nets.MobileUNet import MobileUNet

checkpoint_path = './artifacts/checkpoint_weights.{epoch:02d}-{val_loss:.2f}.h5'
trained_model_path = './artifacts/model5p.h5'

nb_train_samples = 2136
nb_validation_samples = 534

    # transform an onehot array into an array with a gaussian kernel
    # sigma = 1, n = 5
def gaussian(onehot):
    Numofdata = onehot.shape[0]
    for i in range(Numofdata):
        print('gauss: ' + str(i))
        for j in range(5):
            sparse = onehot[i][j]
            for k in range(50176):
                if sparse[k] > 0.01:
                    x = k
            # original point(x,y)
            sparse[x] = 0.195346
            # (x,y+1),(x,y-1),(x+1,y),(x-1,y)
            sparse[x - 224] = sparse[x + 1] = sparse[x - 1] = 0.123317
            if ((x + 224) < len(sparse)):
                sparse[x + 224] = 0.123317
            # (x+1,y+1),(x+1,y-1),(x-1,y+1),(x-1,y-1)
            sparse[x - 223] = sparse[x - 225] = 0.077847

            if ((x + 225) < len(sparse)):
                sparse[x + 225] = 0.077847
                sparse[x + 223] = 0.077847

            # (x,y+2),(x,y-2),(x+2,y),(x-2,y)
            #sparse[x + 448] = sparse[x - 448] = sparse[x + 2] = sparse[x - 2] = 0.023226	
            # (x,y+3),(x,y-3),(x+3,y),(x-3,y) 
            #sparse[x + 672] = sparse[x - 672] = sparse[x + 3] = sparse[x - 3] = 0.002291
            # (x+2,y+1),(x-2,y+1),(x+1,y+2),(x-1,y+2),(x+2,y-1),(x-2,y-1),(x+1,y-2),(x-1,y-2)
            #sparse[x + 226] = sparse[x + 222] = sparse[x + 449] = sparse[x + 447] = sparse[x - 222] = sparse[x - 226] = sparse[x - 447] = sparse[x - 449] = 0.014662
            # (x,y+4),(x,y-4),(x+4,y),(x-4,y) 
            #sparse[x + 896] = sparse[x - 896] = sparse[x + 4] = sparse[x - 4] = 0.002291
            # (x+3,y+1),(x-3,y+1),(x+1,y+3),(x-1,y+3),(x+3,y-1),(x-3,y-1),(x+1,y-3),(x-1,y-3) 
            #sparse[x + 227] = sparse[x + 221] = sparse[x + 673] = sparse[x + 671] = sparse[x - 221] = sparse[x - 227] = sparse[x - 671] = sparse[x - 673] = 0.001446
            # (x+2,y+2),(x+2,y-2),(x-2,y+2),(x-2,y-2)
            #sparse[x + 450] = sparse[x - 446] = sparse[x + 446] = sparse[x - 450] = 0.003676
            # (x+2,y+3),(x-2,y+3),(x+3,y+2),(x-3,y+2),(x+2,y-3),(x-2,y-3),(x+3,y-2),(x-3,y-2)
            #sparse[x + 674] = sparse[x + 670] = sparse[x + 451] = sparse[x + 445] = sparse[x - 670] = sparse[x - 674] = sparse[x - 445] = sparse[x - 451] = 0.000363
            # (x+4,y+1),(x-4,y+1),(x+1,y+4),(x-1,y+4),(x+4,y-1),(x-4,y-1),(x+1,y-4),(x-1,y-4)  
            #sparse[x + 228] = sparse[x + 220] = sparse[x + 897] = sparse[x + 895] = sparse[x - 220] = sparse[x - 228] = sparse[x - 895] = sparse[x - 897] = 0.001794 
            # (x+3,y+3),(x+3,y-3),(x-3,y+3),(x-3,y-3) 
           # sparse[x + 675] = sparse[x - 669] = sparse[x + 669] = sparse[x - 675] = 0.000036 
            # (x+4,y+2),(x-4,y+2),(x+2,y+4),(x-2,y+4),(x+4,y-2),(x-4,y-2),(x+2,y-4),(x-2,y-4) 
            #sparse[x + 452] = sparse[x + 444] = sparse[x + 898] = sparse[x + 894] = sparse[x - 444] = sparse[x - 452] = sparse[x - 894] = sparse[x - 898] = 0.000944 
            # (x+4,y+3),(x-4,y+3),(x+3,y+4),(x-3,y+4),(x+4,y-3),(x-4,y-3),(x+3,y-4),(x-3,y-4) 
            #sparse[x + 676] = sparse[x + 668] = sparse[x + 899] = sparse[x + 893] = sparse[x - 668] = sparse[x - 676] = sparse[x - 893] = sparse[x - 899] = 0.000323 
            # (x+4,y+4),(x+4,y-4),(x-4,y+4),(x-4,y-4) 
            #sparse[x + 900] = sparse[x - 892] = sparse[x + 892] = sparse[x - 900] = 0.000072


          
    return onehot
        
    # data and label are generated here
    # output is (1,224,224,3) and (1,1) array
def generator(data, label, batch_size):
    Numofbatch = data.shape[0]//batch_size
    tempindex = np.arange(0,data.shape[0])
    np.random.shuffle(tempindex)
    cpdata = cp.deepcopy(data)
    cplabel = cp.deepcopy(label)
    for j in range(data.shape[0]):
        cpdata[tempindex[j]] = data[j]
        cplabel[tempindex[j]] = label[j]
    while 1:
        for i in range(Numofbatch):
            yield cpdata[i*batch_size:(i+1)*batch_size],cplabel[i*batch_size:(i+1)*batch_size]
    
    
def train(mobilenet_weights_path, epochs, batch_size):

    # Load training and eval data
    #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    #train_data = mnist.train.images  # Returns np.array
    #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    #eval_data = mnist.test.images  # Returns np.array
    #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)tensorflow export model for 	    inference with batch size 1

    #file containing the path to images and labels
    filename = './dataset.txt'
    filenames = []
    labels1 = []
    labels2 = []
    labels3 = []
    labels4 = []
    labels5 = []
    #reading file and extracting path and labels
    with open(filename, 'r') as File:
        infoFile = File.readlines() #reading lines from files
        for line in infoFile: #reading line by line
            words = line.split(' ')
            filenames.append(words[0])
            tempy1=int(int(words[2]))
            tempx1=int(int(words[1]))
            tempy2=int(int(words[4]))
            tempx2=int(int(words[3]))
            tempy3=int(int(words[6]))
            tempx3=int(int(words[5]))
            tempy4=int(int(words[8]))
            tempx4=int(int(words[7]))
            tempy5=int(int(words[10]))  # label for the 5th point
            tempx5=int(int(words[9]))	# label for the 5th point
            #print(words[0])
            labels1.append(tempy1*224+tempx1)
            labels2.append(tempy2*224+tempx2)
            labels3.append(tempy3*224+tempx3)
            labels4.append(tempy4*224+tempx4)
            labels5.append(tempy5*224+tempx5)
            #labels.append(int(words[1]))
            #labels.append(int(words[2]))
            #labels.append(int(words[3]))
            #labels.append(int(words[4]))
            #labels.append(int(words[5]))
            #labels.append(int(words[6]))
            #labels.append(int(words[7]))
            #labels.append(int(words[8]))
            #labels.append(int(words[9]))
            #labels.append(int(words[10]))

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
    #0.8 train data, 0.2 validation data
    print(NumFiles) 
    i=0
    #labels1 =np.reshape(labels1, (-1,1))
    #labels2 =np.reshape(labels2, (-1,1))
    #labels3 =np.reshape(labels3, (-1,1))
    #labels4 =np.reshape(labels4, (-1,1))
    #labels5 =np.reshape(labels5, (-1,1))
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
        #print(filenames[i])
        imageO = np.cast['f'](imageO)
        imageO = imresize(imageO, (224,224))
        #input_image = np.reshape(imageO, (150528))
        if i < NumFiles*0.8:
            lbl_train_array.append(lbl)
            img_train_array.append(imageO)
        else:
            lbl_val_array.append(lbl)
            img_val_array.append(imageO)
    # convert list to array
    # convert label array to onehot array

    train_data = (np.asarray(img_train_array))
    train_labels = (np.asarray(lbl_train_array))
    print('shape of labels is'+str(train_labels.shape))
    sess = tf.InteractiveSession()
    train_onehot_labels = one_hot(indices=cast(train_labels, dtype='int32'), num_classes=224*224).eval()
    train_gaussian_labels = gaussian(train_onehot_labels)
    train_generator = generator(train_data, train_gaussian_labels, batch_size)
    #train_generator = generator(train_data, train_onehot_labels, batch_size)

    val_data = (np.asarray(img_val_array))
    val_labels = (np.asarray(lbl_val_array))
    val_onehot_labels = one_hot(indices=cast(val_labels, dtype='int32'), num_classes=224*224).eval()
    val_gaussian_labels = gaussian(val_onehot_labels)
    val_generator = generator(val_data, val_gaussian_labels, batch_size)
    #val_generator = generator(val_data, val_onehot_labels, batch_size)
    #train_data = np.transpose(np.asarray(img_array))
    #train_labels = np.transpose(np.asarray(lbl_array))
    print(train_data.shape)
    print(train_onehot_labels.shape)

    lr_base = 0.01 * (float(batch_size) / 16)

    #model is bulit here
    
    prev_model = MobileUNet(input_shape=(224, 224, 3), alpha_up=1)
    top_model = models.Sequential()
    top_model.add(Reshape((5, 224 * 224 *1), input_shape=(None, 224, 224, 5),name='softmax_tensor'))
    model = Model(inputs=prev_model.input, outputs=top_model(prev_model.output))

    # model = MobileDeepLab(input_shape=(img_height, img_width, 3))
    model.load_weights(os.path.expanduser(mobilenet_weights_path.format(224)), by_name=True)
    model = multi_gpu_model(model, gpus=2)
    # Freeze above conv_dw_12
    for layer in model.layers[:70]:
        layer.trainable = True

    # Freeze above conv_dw_13
    # for layer in model.layers[:76]:
    #     layer.trainable = False

    model.summary()
    #model_multi = multi_gpu_model(model, gpus=4)
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        #optimizer=Adam(lr=0.0001),
        loss=softmax_loss,
        metrics=[
            recall,
            precision,
            'accuracy',
        ],
    )

    # callbacks
    scheduler = callbacks.LearningRateScheduler(
        create_lr_schedule(epochs, lr_base=lr_base, mode='progressive_drops'))
    tensorboard = callbacks.TensorBoard(log_dir='./logs')
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=True,
                                           save_best_only=True)

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[scheduler, tensorboard, checkpoint],
    )

    model.save(trained_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mobilenet_weights_path',
        type=str,
        default='./artifacts/mobilenet_1_0_128_tf.h5',
        help='mobilenet weights using imagenet which is available at keras page'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
    )
    args, _ = parser.parse_known_args()

    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    train(**vars(args))
