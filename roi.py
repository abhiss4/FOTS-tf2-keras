import sys
from stn import spatial_transformer_network as transformer
sys.path.append("..")

import tensorflow as tf

# import icdar



def roi_rotate_tensor_pad(feature_map, transform_matrixs, box_masks, box_widths):
		
			max_width = box_widths[tf.argmax(box_widths, 0, output_type=tf.int32)]
			# box_widths = tf.cast(box_widths, tf.float32)
			tile_feature_maps = []
			# crop_boxes = []
			# crop_sizes = []
			# box_inds = []
			map_shape = tf.shape(feature_map)
			map_shape = tf.cast(map_shape, tf.float32)

			for i, mask in enumerate(box_masks): # box_masks is a list of num of rois in each feature map
				_feature_map = feature_map[i]
				# _crop_box = tf.constant([0, 0, 8/map_shape[0], box_widths[i]/map_shape[1]])
				# _crop_size = tf.constant([8, tf.cast(box_widths[i], tf.int32)])
				_feature_map = tf.expand_dims(_feature_map, axis=0)
				box_nums = tf.shape(mask)[0]
				_feature_map = tf.tile(_feature_map, [box_nums, 1, 1, 1])
				# crop_boxes.append(_crop_box)
				# crop_sizes.append(_crop_size)
				tile_feature_maps.append(_feature_map)
				# box_inds.append(i)

			tile_feature_maps = tf.concat(tile_feature_maps, axis=0) # N' * H * W * C where N' = N * B
			trans_feature_map = transformer(tile_feature_maps, transform_matrixs)

			box_nums = tf.shape(box_widths)[0]
			pad_rois = tf.TensorArray(tf.float32, box_nums)
			i = 0

			def cond(pad_rois, i):
				return i < box_nums
			def body(pad_rois, i):
				_affine_feature_map = trans_feature_map[i]
				width_box = box_widths[i]
				# _affine_feature_map = tf.expand_dims(_affine_feature_map, 0)
				# roi = tf.image.crop_and_resize(after_transform, [[0, 0, 8/map_shape[0], width_box/map_shape[1]]], [0], [8, tf.cast(width_box, tf.int32)])
				roi = tf.image.crop_to_bounding_box(_affine_feature_map, 0, 0, 8, width_box)
				pad_roi = tf.image.pad_to_bounding_box(roi, 0, 0, 8, 32)
				pad_rois = pad_rois.write(i, pad_roi)
				i += 1

				return pad_rois, i
			pad_rois, _ = tf.while_loop(cond, body, loop_vars=[pad_rois, i])
			pad_rois = pad_rois.stack()


			return pad_rois
