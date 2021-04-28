import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
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
rdir='C:\\Users\\Abhi\Desktop\\CS2\\icdar2015\\ch4_training_images'
icdar_2015=return_names(rdir,'JPG')
rdir_test='C:\\Users\\Abhi\Desktop\\CS2\\icdar2015\\ch4_test_images\\'
icdar_2015_test=return_names(rdir_test,'JPG')


g='C:\\Users\\Abhi\Desktop\\CS2\\icdar2015\\ch4_training_localization_transcription_gt\\'
icdar_2015_gt=return_names(g,'txt')
g='C:\\Users\\Abhi\Desktop\\CS2\\icdar2015\\Challenge4_Test_Task1_GT\\'
icdar_2015_gt_test=return_names(g,'txt')


train=pd.concat([icdar_2015,icdar_2015_gt],axis=1)
list1=[]
for i in train['txt']:
    f = open(i, "r",encoding='utf-8')
    list1.append(f.readlines())
train['y']=list1
list5=[]
for i in train ['y']:
    list2=[]
    list3=[]
    for j in range(len(i)):
        if j ==0:
            list2.append(                      i[j][1:].split(',')[:-1])
          
            list3.append(i[j].split(',')[-1].replace('\n',''))
    
        else:
             list2.append(        i[j].split(',')[:-1])
             list3.append(i[j].split(',')[-1].replace('\n',''))
    list5.append((list2,list3))
train['label']=[i[1] for i in list5]
train['bb']=[i[0] for i in list5]
train[['label','bb']].head()



#*********************************************************************
def generator_text(df, batch_size=12, random_scale=np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2]),):
    
    count = 1
    images_with_label={}
    l=[]
    images=[]
    gt=np.array(train['bb'])

    index=range(train.shape[0])
    for i in tqdm(index):
        
            im_fn = np.array(train['JPG'])[i]
            im=cv2.imread(im_fn)
            vertexlist,w,h=gt[i],im.shape[1],im.shape[0]
               
            list2=[]
            for j in range(len(vertexlist)):
                list1=[]
                k=0
                while k<7:
                    list1.append((float(vertexlist[j][k]),float(vertexlist[j][k+1])))
                    k=k+2
                list2.append(list1)
               
                

            text_tags=[]
            labels=[]
            for label in train['label'][i]:
                if label == '*' or label == '###' or label == '':
                        text_tags.append(True)
                        labels.append([-1])
                else:
                        labels.append(label_to_array(label))
                        text_tags.append(False)
#                 poly, text_tags, labels=load_annoataion(train['txt'][i])#
            poly=np.array(list2, dtype=np.float32)
            text_tags=np.array(text_tags)
               
            text_polys, text_tags, text_labels = check_and_validate_polys(poly, text_tags, labels, (h, w))

        
            rd_scale = np.random.choice(random_scale)
            im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
            text_polys *= rd_scale

        # rotate image from [-10, 10]j
            angle = random.randint(-10, 10)
            flag=True
            while flag:
                im1, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=False)
                count+=1
                if len(selected_poly)==1:
                    flag=False
                elif count>10:
                    
                    flag=False
           
            count=0
       
            if len(selected_poly)==1:
                text_labels = [text_labels[i] for i in selected_poly]
              
                if text_polys.shape[0] == 0 or len(text_labels) == 0 or text_labels[0][0]==-1:
                    continue
                
                resize_h = 60
                resize_w = 64*4
                im = cv2.resize(im, dsize=(resize_w, resize_h),interpolation = cv2.INTER_AREA)
                cv2.imwrite('C:\\Users\\Abhi\\Desktop\\CS2\\icdar2015\Roi\\'+im_fn.split('\\')[-1],im1)
                images.append('C:\\Users\\Abhi\\Desktop\\CS2\\icdar2015\Roi\\'+im_fn.split('\\')[-1])
                l.append(text_labels[0])
            
                count=count+1
    images_with_label['labels']=l
    images_with_label['im']=images
    return images_with_label



list1=generator_text(train)
Icdar_recog=pd.DataFrame(list1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Icdar_recog['im'].values,Icdar_recog['labels'].values, test_size=0.01, random_state=42)
list_y_train=[]
list_y_test=[]
for x in y_train:
    if len(x)<10:
        x.extend([0]*(10-len(x)))
    elif len(x)>10:
        x=x[:10]
    list_y_train.append(x)
for x in y_test:
    if len(x)<10:
        x.extend([0]*(10-len(x)))
    elif len(x)>10:
        x=x[:10]
    list_y_test.append(x)
    
    
def data_generator_for_recognition(x_data,y_data,batch_size=5):
    batch=0
    images=[]
    labels=[]
    for i,j in zip(x_data,y_data):
        try:
            i=i.decode('utf-8')
            im=cv2.imread(i,0)
            im = cv2.resize(im, dsize=(60, 256),interpolation = cv2.INTER_AREA)
            ret,thr = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
            im=thr[:,:,np.newaxis]
            
#             img = tf.io.read_file(i)
#     # 2. Decode and convert to grayscale
#             img = tf.io.decode_png(img, channels=1)
#     # 3. Convert to float32 in [0, 1] range
#             img = tf.image.convert_image_dtype(img, tf.float32)
#     # 4. Resize to the desired size
#             img = tf.image.resize(img, [60, 256])
#     # 5. Transpose the image because we want the time
#     # dimension to correspond to the width of the image.
#             img = tf.transpose(img, perm=[1, 0, 2])
            images.append(im)
            labels.append(j)
            batch+=1
            if batch==batch_size:
                yield np.array(images,dtype=np.float32),np.array(labels,dtype=np.float32)
            
                images=[]
                labels=[]
                batch=0
         
        except Exception as e:
                import traceback
              
                traceback.print_exc()
                continue
            
            
inputs =Input(name='input', shape=(60,256,1), dtype='float32')  


# x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inputs) 
# x = MaxPool2D((2, 2), name="pool1")(x)

#     # Second conv block
# x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x) 
# x = MaxPool2D((2, 2), name="pool2")(x)

# print(x.shape)

conv1 = Conv2D(16, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs) 
bn1 = BatchNormalization()(conv1)
act1 = Activation('relu')(bn1)
x=Dropout(.2)
maxpool1 = MaxPool2D(pool_size=(2, 2), name='max1')(act1)  

conv2 = Conv2D(32, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(maxpool1)  
bn2 = BatchNormalization()(conv2)
act2 = Activation('relu')(bn2)
x=Dropout(.2)
maxpool2 = MaxPool2D(pool_size=(2, 2), name='max2')(act2) 

conv3 = Conv2D(32, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(maxpool2)  
bn3 = BatchNormalization()(conv3)
act3 =Activation('relu')(bn3)
x=Dropout(.2)
conv4 = Conv2D(64, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(act3)  
bn4 = BatchNormalization()(conv4)
act4 =Activation('relu')(bn4)
# maxpool3 = MaxPool2D(pool_size=(2, 2), name='max3')(act4)  
x=Dropout(.2)
conv5 = Conv2D(64, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(act4)
bn5 = BatchNormalization()(conv5)
act5 = Activation('relu')(bn5)
x=Dropout(.2)
conv6 = Conv2D(128, (3, 3), padding='same', name='conv6')(act5)  
bn6 = BatchNormalization()(conv6)
act6 = Activation('relu')(bn6)
print(act6.shape)


conv7 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='con7')(act6) 
print(conv7.shape)
bn7 = BatchNormalization()(conv7)
act7 = Activation('relu')(bn7)

# x = UpSampling2D(size=[2, 2],interpolation='bilinear')(act7)
print(act7.shape)
re1 = Reshape(target_shape=((15,64*128)), name='reshape')(act7)  

dense1 = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(re1) 
x= Dropout(.2)
bidi1=Bidirectional(LSTM(64,return_sequences=True,go_backwards=True))(dense1)
out1=Bidirectional(LSTM(64,return_sequences=True,go_backwards=True))(bidi1)

out2=Dense(97,activation ='softmax')(out1)

recognizer=Model(inputs,out2)
def ctc_loss(y_true,y_pred):
  
  #https://stackoverflow.com/questions/64321779/how-to-use-tf-ctc-loss-with-variable-length-features-and-labels

    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True)
        
    
    return tf.keras.backend.ctc_batch_cost(y_true,y_pred,np.ones((10,1),'int32')*15,label_length)


recognizer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001
                                                 ,amsgrad=True),loss=ctc_loss)




dataset_rec = tf.data.Dataset.from_generator(
     data_generator_for_recognition,
     (tf.float32, tf.float32),
     args=(X_train,list_y_train,10))
dataset_rec = dataset_rec.cache()

tensorboard=tf.keras.callbacks.TensorBoard(log_dir='recog_logs',write_images=True,histogram_freq=1,write_graph=False)
early_stop=tf.keras.callbacks.EarlyStopping(monitor='loss',patience=6,mode='min',verbose=1)
modelchkpt=tf.keras.callbacks.ModelCheckpoint('recog1.h5',save_best_only=True,mode='min',monitor='loss')
g=tf.keras.callbacks.ReduceLROnPlateau(monitor="loss",factor=0.5,patience=3,verbose=True)
callbacks=[tensorboard,early_stop,modelchkpt,g]


recognizer.fit(dataset_rec.repeat(),epochs=100,steps_per_epoch=len(X_train),callbacks=callbacks)