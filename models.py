import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
from pipeline import load_annoataion,generator_synth,get_batch_synth,generator,get_batch,generator_a,generator_i
import cv2
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pathlib
import pandas as pd
import os
from Levenshtein import distance
from enque import GeneratorEnqueuer
from itertools import compress
# from shapely.geometry import Polygon
import time
import random
import threading
import multiprocessing
import config
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from utils import check_and_validate_polys, crop_area, rotate_image, generate_rbox,generate_r, get_project_matrix_and_width, sparse_tuple_from, crop_area_fix

from roi import roi_rotate_tensor_pad as roi
#model related imports
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2,ResNet50,DenseNet121
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.executing_eagerly()
from tensorflow.keras.layers import Dense,Input,Activation,Dropout,Flatten,BatchNormalization,Concatenate,Conv2D,AveragePooling2D,Conv2DTranspose,UpSampling2D,Lambda,Bidirectional,LSTM,MaxPool2D,Reshape
def return_names(root_dir,str1):
    data_root = pathlib.Path(root_dir)
    print("root is",data_root)
    all_pics_path=list(data_root.glob("**/*."+ str1))
    all_pics_path=[str(path) for path in all_pics_path]
    dict1={str1: all_pics_path}

    data_df=pd.DataFrame(dict1)
    return data_df


import tensorflow_addons as tfa
tfa_enabled = False

def label_to_array(label):
    try:
        label = label.replace(' ', '')
        return [CHAR_VECTOR.index(x) for x in label]
    except Exception as ex:
        print(label)
        raise ex
CHAR_VECTOR = "#0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMÉNOPQRSTUVWXYZ-~`´<>'.:;^/|!?$%@&*()[]{}_+=,\\\""
num_classes=len(CHAR_VECTOR)+1
from tqdm import tqdm
from keras.models import load_model


class Detector(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.pred_score_map = Conv2D(1, (1, 1), activation='sigmoid', name='pred_score_map')
        self.rbox_geo_map = Conv2D(4, (1, 1), activation='sigmoid', name='rbox_geo_map')
        self.scale_rbox_geo_map = Lambda(lambda x: x * 512)
        self.angle_map = Conv2D(1, (1, 1), activation='sigmoid', name='rbox_angle_map')
        self.scale_angle_map = Lambda(lambda x: (x - 0.5) * np.pi / 2)
    def call(self,backbone):
        z = self.pred_score_map(backbone)
        x= self.rbox_geo_map(backbone)
        x= self.scale_rbox_geo_map(x)
        
        y=self.angle_map(backbone) 
        y=  self.scale_angle_map(y)
        zxy= Concatenate(axis=-1,name='pred_geo_map')([z,x,y])
        return zxy
    
    
    
def detector():                                
    input_shape=(512,512,3)
    input_layer = Input(shape=input_shape,name = 'image_input')
    net=DenseNet121(input_tensor=input_layer,weights="imagenet",include_top=False,  input_shape=input_shape)
    net.trainable=True
    
    layer=net.get_layer('relu').output
    
    l=['pool4_conv','pool3_conv','pool2_conv']
    layer_2, layer_3, layer_4=[net.get_layer(i).output for i in l]
    
    layer_1 = UpSampling2D(size=[2, 2],interpolation='bilinear')(layer)
    
    x = Conv2D(filters=128, kernel_size=1, padding='same')(tf.concat([layer_1, layer_2], axis=-1))
    x = Conv2D(filters=128, kernel_size=3, padding='same')(x)
    x = BatchNormalization( momentum=0.997, epsilon=0.00001)(x)
    x = Activation('relu')(x)
    x=Dropout(.2)(x)
    x = UpSampling2D(size=[2, 2],interpolation='bilinear')(x)
    
    
    x = Conv2D(filters=64, kernel_size=1, padding='same')(tf.concat([x, layer_3], axis=-1))
    x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
    x = BatchNormalization( momentum=0.997, epsilon=0.00001)(x)
    x = Activation('relu')(x)
    x=Dropout(.2)(x)
    x = UpSampling2D(size=[2, 2],interpolation='bilinear')(x)
    
    
    
    x = Conv2D(filters=32, kernel_size=1, padding='same')(tf.concat([x, layer_4], axis=-1))
    x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
    x = BatchNormalization( momentum=0.997, epsilon=0.00001)(x)
    x = Activation('relu')(x)
    x=Dropout(.2)(x)
    x=Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    
    detector=Detector()
    
    out=detector(x)
    
    d = Model(inputs=[input_layer], outputs=[out])
    return d







def r():
    inputs =Input(name='input', shape=(60,256,1), dtype='float32')  


# x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inputs) 
# x = MaxPool2D((2, 2), name="pool1")(x)

#     # Second conv block
# x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x) 
# x = MaxPool2D((2, 2), name="pool2")(x)

# print(x.shape)

    conv1 = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs) 
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    x=Dropout(.2)
    maxpool1 = MaxPool2D(pool_size=(2, 2), name='max1')(act1)  
    
    conv2 = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(maxpool1)  
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)
    x=Dropout(.2)
    maxpool2 = MaxPool2D(pool_size=(2, 2), name='max2')(act2) 
    
    conv3 = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(maxpool2)  
    bn3 = BatchNormalization()(conv3)
    act3 =Activation('relu')(bn3)
    x=Dropout(.2)
    conv4 = Conv2D(128, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(act3)  
    bn4 = BatchNormalization()(conv4)
    act4 =Activation('relu')(bn4)
    # maxpool3 = MaxPool2D(pool_size=(2, 2), name='max3')(act4)  
    x=Dropout(.2)
    conv5 = Conv2D(128, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(act4)
    bn5 = BatchNormalization()(conv5)
    act5 = Activation('relu')(bn5)
    x=Dropout(.2)
    conv6 = Conv2D(128, (3, 3), padding='same', name='conv6')(act5)  
    bn6 = BatchNormalization()(conv6)
    act6 = Activation('relu')(bn6)
    
    
    
    conv7 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name='con7')(act6) 
  
    bn7 = BatchNormalization()(conv7)
    act7 = Activation('relu')(bn7)
    
    # x = UpSampling2D(size=[2, 2],interpolation='bilinear')(act7)
  
    re1 = Reshape(target_shape=((64,15*256)), name='reshape')(act7)  
    
    dense1 = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(re1) 
    x= Dropout(.2)
    bidi1=Bidirectional(LSTM(128,return_sequences=True,go_backwards=True))(dense1)
    out1=Bidirectional(LSTM(128,return_sequences=True,go_backwards=True))(bidi1)
    
    out2=Dense(97,activation ='softmax')(out1)
    
    recognizer=Model(inputs,out2)
    return recognizer
