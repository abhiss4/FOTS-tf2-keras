import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pathlib
import pandas as pd





import os




import cv2

from utils import check_and_validate_polys, crop_area, rotate_image, generate_rbox,generate_r, get_project_matrix_and_width, sparse_tuple_from, crop_area_fix
from enque import GeneratorEnqueuer
import tensorflow as tf

from itertools import compress
# from shapely.geometry import Polygon
import time
import random

def return_names(root_dir,str1):
    data_root = pathlib.Path(root_dir)
    print("root is",data_root)
    all_pics_path=list(data_root.glob("**/*."+ str1))
    all_pics_path=[str(path) for path in all_pics_path]
    dict1={str1: all_pics_path}

    data_df=pd.DataFrame(dict1)
    return data_df

CHAR_VECTOR = "#0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-~`´<>'.:;^/|!?$%@&*()[]{}_+=,\\\""
def label_to_array(label):
    try:
        label = label.replace(' ', '')
        return [CHAR_VECTOR.index(x) for x in label]
    except Exception as ex:
        print(label)
        raise ex

def generator(train, input_size=512, batch_size=12,min_text_size=10):
    # data_loader = SynthTextLoader()
   
    # image_list = np.array(data_loader.get_images(FLAGS.training_data_dir))
    image_list = np.array(train['JPG'])
    gt=np.array(train['bb'])
    # print('{} training images in {} '.format(image_list.shape[0], FLAGS.training_data_dir))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        batch_images = []
        batch_image_fns = []
        batch_score_maps = []
        batch_geo_maps = []
        batch_training_masks = []

        batch_text_polyses = [] 
        batch_text_tagses = []
        batch_boxes_masks = []

        batch_text_labels = []
        rboxes = []
        count = 0
        for i in index:
            try:
                im_fn = image_list[i]
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
                poly=np.array(list2)
                text_tags=np.array(text_tags)
               
                poly, text_tags, text_labels = check_and_validate_polys(poly, text_tags, labels, (h, w))

                ############################# Data Augmentation ##############################
                # random scale this image
               
                scale=random.choice((.8,.9,1.0,1.1,1.2))
                im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
                text_polys = poly*scale
              
#                 rotate image from [-10, 10]
                angle=random.randint(-10,10)
                im, text_polys = rotate_image(im, text_polys, angle)
               
                # 600×600 random samples are cropped.
                im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=False)
                # im, text_polys, text_tags, selected_poly = crop_area_fix(im, text_polys, text_tags, crop_size=(600, 600))
                text_labels = [text_labels[i] for i in selected_poly]
                if text_polys.shape[0] == 0 or len(text_labels) == 0:
                    continue

                # pad the image to the training input size or the longer side of image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = im_padded
                # resize the image to input size
                new_h, new_w, _ = im.shape
                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize=(resize_w, resize_h),interpolation = cv2.INTER_AREA)
                resize_ratio_3_x = resize_w/float(new_w)
                resize_ratio_3_y = resize_h/float(new_h)
                text_polys[:, :, 0] *= resize_ratio_3_x
                text_polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape
                score_map, geo_map, training_mask,rbox,rectangles = generate_r((new_h, new_w), text_polys, text_tags,min_text_size)
                #mask = [not (word == [-1]) for word in text_labels]
                #text_labels = list(compress(text_labels, mask))
                #rectangles = list(compress(rectangles, mask))
                mask1 = [not (word == [-1]) for word in text_labels]
                mask2 = [j[1] > min_text_size and j[0] > min_text_size for j in [i[2::] for i in rbox[1]]]  # make sure the text is at least 4x4 pixels
                mask = [True if i and j else False for (i, j) in zip(mask1, mask2)]
                text_labels = list(compress(text_labels, mask))
                rectangles = list(compress(rectangles, mask))
                rbox = tuple([list(compress(item, mask)) for item in rbox])
               
              
                assert len(text_labels) == len(rectangles), "rotate rectangles' num is not equal to text label"

                if len(text_labels) == 0:
                    continue

                boxes_mask = np.array([count] * len(rectangles))

                count += 1

                batch_images.append(im[:, :, ::-1].astype(np.float32))
              
                batch_image_fns.append(im_fn)
                batch_score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                batch_geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                batch_training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))
              
                batch_text_polyses.append(rectangles)
                batch_boxes_masks.append(boxes_mask)
                batch_text_labels.extend(text_labels)
                batch_text_tagses.append(text_tags)
                rboxes.append(rbox)
                if len(batch_images) == batch_size:
                    batch_text_polyses = np.concatenate(batch_text_polyses)
                    batch_text_tagses = np.concatenate(batch_text_tagses)
                    batch_transform_matrixes, batch_box_widths = get_project_matrix_and_width(batch_text_polyses, batch_text_tagses)
                   
                    batch_text_labels_sparse = sparse_tuple_from(np.array(batch_text_labels))

                    # yield images, image_fns, score_maps, geo_maps, training_masks
                    yield np.array(batch_images), batch_image_fns, np.array(batch_score_maps), np.array(batch_geo_maps), np.array(batch_training_masks), batch_transform_matrixes, batch_boxes_masks, batch_box_widths, batch_text_labels_sparse, batch_text_polyses, batch_text_labels,rboxes
                    batch_images = []
                    batch_image_fns = []
                    batch_score_maps = []
                    batch_geo_maps = []
                    batch_training_masks = []
                    batch_text_polyses = [] 
                    batch_text_tagses = []
                    batch_boxes_masks = []
                    batch_text_labels = []
                    rboxes=[]
                    count = 0
            except Exception as e:
                import traceback
              
                traceback.print_exc()
                continue
            
            
def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=False)
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()
            




def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    # print p
    text_polys = []
    text_tags = []
    labels = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r', encoding='utf-8-sig') as f:
        for line in f.readlines():
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            # line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            line = line.replace('\xef\xbb\bf', '')
            line = line.replace('\xe2\x80\x8d', '')
            line = line.strip()
            line = line.split(' ')
            if len(line) > 9:
                label = line[8]
                for i in range(len(line) - 9):
                    label = label + "," + line[i + 9]
            else:
                label = line[-1]
            # label = line[-1]
            line = [line[0]] + [line[4]] + [line[1]] + [line[5]] + [line[2]] + [line[6]] + [line[3]] + [line[7]]
            temp_line = map(eval, line[:8])
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, temp_line)
            # x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###' or label == '':
                text_tags.append(True)
                labels.append([-1])
            else:
                labels.append(label_to_array(label))
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool), labels


def generator_synth(image_list, input_size=512, batch_size=12):
    
        
    index = np.arange(0, len(image_list))
    while True:
        np.random.shuffle(index)
        batch_images = []
        batch_image_fns = []
        batch_score_maps = []
        batch_geo_maps = []
        batch_training_masks = []

        batch_text_polyses = [] 
        batch_text_tagses = []
        batch_boxes_masks = []
        rboxes=[]
        batch_text_labels = []
        count = 0
        for i in index:
            try:
                im_fn = image_list[i]
                im=cv2.imread(im_fn)
                
                poly, text_tags, labels=load_annoataion('C:\\Users\\Abhi\\Desktop\\CS2\\SynthText\\annotation\\'+image_list[i].split('/')[1].split('.')[0]+'.txt')
            
                h,w=im.shape[0],im.shape[1]
                poly, text_tags, text_labels = check_and_validate_polys(poly, text_tags, labels, (h, w))

                ############################# Data Augmentation ##############################
                # random scale this image
               
                scale=random.choice((.8,.9,1.0,1.1,1.2))
                im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
                text_polys = poly*scale
              
#                 rotate image from [-10, 10]
                angle=random.randint(-10,10)
                im, text_polys = rotate_image(im, text_polys, angle)
               
                # 600×600 random samples are cropped.
                im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=False)
                # im, text_polys, text_tags, selected_poly = crop_area_fix(im, text_polys, text_tags, crop_size=(600, 600))
                text_labels = [text_labels[i] for i in selected_poly]
                if text_polys.shape[0] == 0 or len(text_labels) == 0:
                    continue

                # pad the image to the training input size or the longer side of image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = im_padded
                # resize the image to input size
                new_h, new_w, _ = im.shape
                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize=(resize_w, resize_h),interpolation = cv2.INTER_AREA)
                resize_ratio_3_x = resize_w/float(new_w)
                resize_ratio_3_y = resize_h/float(new_h)
                text_polys[:, :, 0] *= resize_ratio_3_x
                text_polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape
                score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)
                mask = [not (word == [-1]) for word in text_labels]
                text_labels = list(compress(text_labels, mask))
                rectangles = list(compress(rectangles, mask))
 

      
              
                assert len(text_labels) == len(rectangles), "rotate rectangles' num is not equal to text label"

                if len(text_labels) == 0:
                    continue

                boxes_mask = np.array([count] * len(rectangles))

                count += 1

                batch_images.append(im)
              
                batch_image_fns.append(im_fn)
                batch_score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                batch_geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                batch_training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))
               
                batch_text_polyses.append(rectangles)
                batch_boxes_masks.append(boxes_mask)
                batch_text_labels.extend(text_labels)
                batch_text_tagses.append(text_tags)
                if len(batch_images) == batch_size:
                    batch_text_polyses = np.concatenate(batch_text_polyses)
                    batch_text_tagses = np.concatenate(batch_text_tagses)
                    batch_transform_matrixes, batch_box_widths = get_project_matrix_and_width(batch_text_polyses, batch_text_tagses)
                   
                    batch_text_labels_sparse = sparse_tuple_from(np.array(batch_text_labels))

                    # yield images, image_fns, score_maps, geo_maps, training_masks
                    yield np.array(batch_images), batch_image_fns, np.array(batch_score_maps), np.array(batch_geo_maps), np.array(batch_training_masks), batch_transform_matrixes, batch_boxes_masks, batch_box_widths, batch_text_labels_sparse, batch_text_polyses, batch_text_labels,rboxes
                    batch_images = []
                    batch_image_fns = []
                    batch_score_maps = []
                    batch_geo_maps = []
                    batch_training_masks = []
                    batch_text_polyses = [] 
                    batch_text_tagses = []
                    batch_boxes_masks = []
                    batch_text_labels = []
                    rboxes=[]
                    count = 0
            except Exception as e:
                import traceback
              
                traceback.print_exc()
                continue
            
def get_batch_synth(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator_synth(**kwargs), use_multiprocessing=False)
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()
            
            
            
            
            
            
            

def generator_i(train, input_size=512, batch_size=1):
    # data_loader = SynthTextLoader()
   
    # image_list = np.array(data_loader.get_images(FLAGS.training_data_dir))
    image_list = np.array(train['JPG'])
    gt=np.array(train['bb'])
    # print('{} training images in {} '.format(image_list.shape[0], FLAGS.training_data_dir))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        batch_images = []
        batch_image_fns = []
        batch_score_maps = []
        batch_geo_maps = []
        batch_training_masks = []

    
    
        count = 0
        for i in index:
            try:
                im_fn = image_list[i]
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
                poly=np.array(list2)
                text_tags=np.array(text_tags)
               
                text_polys, text_tags, text_labels = check_and_validate_polys(poly, text_tags, labels, (h, w))

                ############################# Data Augmentation ##############################
                # random scale this image
               
#                 scale=random.choice((.8,.9,1.0,1.1,1.2))
#                 im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
#                 text_polys = poly*scale
              
# #                 rotate image from [-10, 10]
#                 angle=random.randint(-10,10)
#                 im, text_polys = rotate_image(im, text_polys, angle)
               
#                 # 600×600 random samples are cropped.
#                 im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=False)
#                 # im, text_polys, text_tags, selected_poly = crop_area_fix(im, text_polys, text_tags, crop_size=(600, 600))
#                 text_labels = [text_labels[i] for i in selected_poly]
#                 if text_polys.shape[0] == 0 or len(text_labels) == 0:
#                     continue

#                 # pad the image to the training input size or the longer side of image
#                 new_h, new_w, _ = im.shape
#                 max_h_w_i = np.max([new_h, new_w, input_size])
#                 im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
#                 im_padded[:new_h, :new_w, :] = im.copy()
#                 im = im_padded
                # resize the image to input size
                new_h, new_w, _ = im.shape
                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize=(resize_w, resize_h),interpolation = cv2.INTER_AREA)
                resize_ratio_3_x = resize_w/float(new_w)
                resize_ratio_3_y = resize_h/float(new_h)
                text_polys[:, :, 0] *= resize_ratio_3_x
                text_polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape
                score_map, geo_map, training_mask,rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)
                mask = [not (word == [-1]) for word in text_labels]
                text_labels = list(compress(text_labels, mask))
                rectangles = list(compress(rectangles, mask))

               
              
                assert len(text_labels) == len(rectangles), "rotate rectangles' num is not equal to text label"

                if len(text_labels) == 0:
                    continue


                count += 1

                batch_images.append(im[:, :, ::-1].astype(np.float32))
              
                batch_image_fns.append(im_fn)
                batch_score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                batch_geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                batch_training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))
#                 rboxes.append(rbox)
#                 batch_text_polyses.append(rectangles)
#                 batch_boxes_masks.append(boxes_mask)
#                 batch_text_labels.extend(text_labels)
#                 batch_text_tagses.append(text_tags)
                if len(batch_images) == batch_size:
#                     batch_text_polyses = np.concatenate(batch_text_polyses)
#                     batch_text_tagses = np.concatenate(batch_text_tagses)
#                     batch_transform_matrixes, batch_box_widths = get_project_matrix_and_width(batch_text_polyses, batch_text_tagses)
                   
#                     batch_text_labels_sparse = sparse_tuple_from(np.array(batch_text_labels))

                    # yield images, image_fns, score_maps, geo_maps, training_masks
                    yield  (tf.keras.applications.densenet.preprocess_input(np.array(batch_images)),np.concatenate([np.array(batch_score_maps), np.array(batch_geo_maps), np.array(batch_training_masks)],axis=3))
                    batch_images = []
                    batch_image_fns = []
                    batch_score_maps = []
                    batch_geo_maps = []
                    batch_training_masks = []
                  
                    count = 0
            except Exception as e:
                import traceback
              
                traceback.print_exc()
                continue




#****************************************************************************************************************************


def generator_a(image_list, input_size=512, batch_size=1):
    
        
    index = np.arange(0, len(image_list))
    while True:
        np.random.shuffle(index)
        batch_images = []
        batch_image_fns = []
        batch_score_maps = []
        batch_geo_maps = []
        batch_training_masks = []

#         batch_text_polyses = [] 
#         batch_text_tagses = []
#         batch_boxes_masks = []
#         rboxes=[]
#         batch_text_labels = []
        count = 0
        for i in index:
            try:
                im_fn = image_list[i]
                im=cv2.imread(im_fn)
                
                poly, text_tags, labels=load_annoataion('C:\\Users\\Abhi\\Desktop\\CS2\\SynthText\\annotation\\'+image_list[i].split('/')[1].split('.')[0]+'.txt')
            
                h,w=im.shape[0],im.shape[1]
                text_polys, text_tags, text_labels = check_and_validate_polys(poly, text_tags, labels, (h, w))

                ############################# Data Augmentation ##############################
                # random scale this image
               
#                 scale=random.choice((.8,.9,1.0,1.1,1.2))
#                 im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
#                 text_polys = poly*scale
              
# #                 rotate image from [-10, 10]
#                 angle=random.randint(-10,10)
#                 im, text_polys = rotate_image(im, text_polys, angle)
               
# #                600×600 random samples are cropped.
#                 im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=False)
#                 # im, text_polys, text_tags, selected_poly = crop_area_fix(im, text_polys, text_tags, crop_size=(600, 600))
#                 text_labels = [text_labels[i] for i in selected_poly]
#                 if text_polys.shape[0] == 0 or len(text_labels) == 0:
#                     continue

#                 # pad the image to the training input size or the longer side of image
#                 new_h, new_w, _ = im.shape
#                 max_h_w_i = np.max([new_h, new_w, input_size])
#                 im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
#                 im_padded[:new_h, :new_w, :] = im.copy()
#                 im = im_padded
                # resize the image to input size
                new_h, new_w, _ = im.shape
            
                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize=(resize_w, resize_h),interpolation = cv2.INTER_AREA)
         
                resize_ratio_3_x = resize_w/float(new_w)
                resize_ratio_3_y = resize_h/float(new_h)
                text_polys[:, :, 0] *= resize_ratio_3_x
                text_polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape
               
                score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)
#                 mask = [not (word == [-1]) for word in text_labels]
#                 text_labels = list(compress(text_labels, mask))
#                 rectangles = list(compress(rectangles, mask))
                if type(score_map)!=np.ndarray:
                    if score_map==None:
                        continue

      
              
#                 assert len(text_labels) == len(rectangles), "rotate rectangles' num is not equal to text label"

#                 if len(text_labels) == 0:
#                     continue

#                 boxes_mask = np.array([count] * len(rectangles))

                count += 1

                batch_images.append(im)
              
                batch_image_fns.append(im_fn)
                batch_score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                batch_geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                batch_training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))
               
#                 batch_text_polyses.append(rectangles)
#                 batch_boxes_masks.append(boxes_mask)
#                 batch_text_labels.extend(text_labels)
#                 batch_text_tagses.append(text_tags)
                if len(batch_images) == batch_size:
#                     batch_text_polyses = np.concatenate(batch_text_polyses)
#                     batch_text_tagses = np.concatenate(batch_text_tagses)
#                     batch_transform_matrixes, batch_box_widths = get_project_matrix_and_width(batch_text_polyses, batch_text_tagses)
                   
#                     batch_text_labels_sparse = sparse_tuple_from(np.array(batch_text_labels))
                    yield (tf.keras.applications.densenet.preprocess_input(np.array(batch_images)),np.concatenate([np.array(batch_score_maps), np.array(batch_geo_maps), np.array(batch_training_masks)],axis=3))
                    # yield images, image_fns, score_maps, geo_maps, training_masks
#                     yield np.array(batch_images), batch_image_fns, np.array(batch_score_maps), np.array(batch_geo_maps), np.array(batch_training_masks), batch_transform_matrixes, batch_boxes_masks, batch_box_widths, batch_text_labels_sparse, batch_text_polyses, batch_text_labels,rboxes
                    batch_images = []
                    batch_image_fns = []
                    batch_score_maps = []
                    batch_geo_maps = []
                    batch_training_masks = []
        
                    count = 0
            except Exception as e:
                import traceback
              
                traceback.print_exc()
                continue


