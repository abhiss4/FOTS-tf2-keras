import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pathlib
import pandas as pd
import time
import cv2
import tensorflow as tf

import streamlit as st
from PIL import Image
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import warnings
warnings.filterwarnings("ignore")

# from shapely.geometry import Polygon


#model related imports
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
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




CHAR_VECTOR = "#0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMÉNOPQRSTUVWXYZ-~`´<>'.:;^/|!?$%@&*()[]{}_+=,\\\""
num_classes=len(CHAR_VECTOR)+1

from models import detector,r

from inf_fns import restore_rectangle,nms_locality,restore_roiRotatePara
import streamlit as st


# recognizer.load_weights('recog.h5')

def Inference(path,d,recognizer):
    im=Image.open(path)
    im=np.array(im)
    im=cv2.resize(im, dsize=(512, 512),interpolation = cv2.INTER_AREA)

    
    
    pred=d.predict(im[np.newaxis,:,:,:])
    score_map=pred[0][:,:,0]
    
    geo_map=pred[0][:,:,1:]
    for ind in [0,1,2,3,4]:
        geo_map[:,:,ind]*=score_map
    score_map_thresh=0.2
    box_thresh=0.3
    nms_thres=0.05
    if len(score_map.shape) == 4:
   
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, :]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)

    # print(xy_text)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :])
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    boxes = nms_locality(boxes.astype(np.float64), nms_thres)
    # boxes = cv2.dnn.NMSBoxes(boxes, score_map, 0.3, (0.1))

    res = []
    result = []
    if len(boxes)>0:
        for box in boxes:
            box_ =  box[:8].reshape((4, 2))
            if np.linalg.norm(box_[0] - box_[1]) < 8 or np.linalg.norm(box_[3]-box_[0]) < 8:
                continue
            result.append(box_)
    res.append(np.array(result, np.float32))   
    #print(res)
    box_index = []
    brotateParas = []
    filter_bsharedFeatures = []
    for i in range(len(res)):
        rotateParas = []
        rboxes=res[i]
        txt=[]
        for j, rbox in enumerate(rboxes):
            # print(rbox)
            para = restore_roiRotatePara(rbox)
            #break
            if para and min(para[1][2:]) > 8:
                rotateParas.append(para)
                box_index.append((i, j))
        pts=[] 

    if len(rotateParas) > 0:
        for num in range(len(rotateParas)):
            text=""
            out=[]
            out1=rotateParas[num][0]
            for f in out1 :
            
                if f <0:
                    f=-1*f
                out.append(f)
        
            crop=rotateParas[num][1]
            points=np.array([[out[0],out[1]],[out[0]+out[2],out[1]],[out[0]+out[2],out[1]+out[3]],[out[0],out[1]+out[3]]])
            angle=rotateParas[num][2] 
                #print(out)
#                 img1=tf.image.crop_to_bounding_box(img,out[1]-(int(out[1]*(5/100))),out[0]-(int(out[0]*(5/100))),out[3]+(int(out[3]*(55/100))),out[2]+(int(out[2]*(55/100))))
                # print(out)
            img1=tf.image.crop_to_bounding_box(im,out[1],out[0],out[3],out[2])
                #print(img1.shape)
#                 plt.imshow(img1)
#                 plt.show()
            img2=tf.keras.preprocessing.image.random_rotation(img1,angle)
                
                #print(crop)
                #print(crop[0])
            img2=tf.image.crop_to_bounding_box(img2,crop[1],crop[0],crop[3],crop[2]).numpy()
#                 plt.imshow(img2)
#                 plt.show()
            img2=cv2.resize(img2,(256,60))
            img2=cv2.detailEnhance(img2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ret,thr = cv2.threshold(img2, 0, 255, cv2.THRESH_OTSU)
            pred=recognizer.predict(thr[np.newaxis,:,:,np.newaxis])
            pred_arr=tf.keras.backend.ctc_decode(pred,np.ones((1),'int8')*64,)

            x=''
            for i in pred_arr[0][0][0]:
    
                if i== -1:
                    continue
                else:
                    x+=''.join(CHAR_VECTOR[i])
            txt.append(x)
            pts.append(points)  
    return pts,txt,im


st.set_option('deprecation.showfileUploaderEncoding', False)
st.write(" # FAST ORIENTED SCENE TEXT SPOTTING ")

@st.cache
def load_model():
      return detector(),r()
        

with st.spinner('Hang tight loading the model')  :
    d,recognizer=load_model()
    d.load_weights('detector_model.h5'),
    recognizer.load_weights('recog.h5')
     
#**********************************************************************
path = st.file_uploader("Choose an image...", type="jpg")

if path is None:
    st.text('Please upload an image')
else:
   start_time = time.time()

   pts,txt,im= Inference(path,d,recognizer)
   for i in range(len(txt)):
        cv2.polylines(im,[pts[i]],isClosed=True,color=(255,0,0),thickness=1)
        cv2.putText(im,txt[i],(pts[i][0][0],pts[i][0][1]),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 1)
      
  

   
    
   st.image(im)
   end_time = time.time()
   st.write("Time Taken By Pipeline=" + str(end_time - start_time) + " seconds")
   
    