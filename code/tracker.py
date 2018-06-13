import pandas as pd
import numpy as np
import cv2 as cv
from collections import namedtuple
import uuid

def calc_overlap_area(bbox1, bbox2):
  Bbox = namedtuple('Rectangle', 'xmin ymin xmax ymax')
  input1 = Bbox(bbox1[1], bbox1[0], bbox1[3], bbox1[2])
  input2 = Bbox(bbox2[1], bbox2[0], bbox2[3], bbox2[2])
  dx = min(input1.xmax, input2.xmax) - max(input1.xmin, input2.xmin)
  dy = min(input1.ymax, input2.ymax) - max(input1.ymin, input2.ymin)
  if (dx>=0) and (dy>=0):
    return dx*dy
  return 0

def calc_area(bbox):
  return (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

def calc_i_o_u(bbox1, bbox2):
  intersect = calc_overlap_area(bbox1, bbox2)
  union = calc_area(bbox1) + calc_area(bbox2) - intersect
  return intersect/union

def calc_pixels(box, frame_shape):
  result = []
  result.append(box[0] * frame_shape[0])
  result.append(box[1] * frame_shape[1])
  result.append(box[2] * frame_shape[0])
  result.append(box[3] * frame_shape[1])
  return np.array(result).astype(np.int32)

def assign_obj_id():
  return str(uuid.uuid4())

'''
the bounding box (bbox) property is [starting_y, starting_x, ending_y, ending_x]
the result is an array
'''
def make_frame_info(frame_count, squeezed_boxes, squeezed_classes, squeezed_scores, frame_shape, frame_image):
  result = []
  for i in range(len(squeezed_scores)):
    if squeezed_scores[i] > 0.50:
      bbox = calc_pixels(squeezed_boxes[i], frame_shape)
      result.append({
        'frame_count': frame_count,
        'class': str(squeezed_classes[i]),
        'score': str(squeezed_scores[i]),
        'bbox': bbox,
        'roi_hist': extract_hsv_histogram(frame_image, bbox),
        'obj_id': assign_obj_id()
      })

    else:
      break
  result = pd.DataFrame(result)
  return result

def append_current_to_past_frame_info(past_frames, current_frame):
  if len(past_frames) == 0:
    return pd.DataFrame(current_frame)
  return past_frames.append(current_frame)

def extract_hsv_histogram(frame, bbox):
  roi = frame[bbox[0]:bbox[2], bbox[1]:bbox[3]]
  hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
  mask = cv.inRange(hsv_roi, np.array((0, 60,32.)), np.array((180,255.,255.)))
  roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
  roi_hist = cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
  return roi_hist

def get_matching_objects(last_frame, current_frame, iou_threshold, hist_threshold):
  result_frame = pd.DataFrame([])
  for class_id in current_frame['class']:
    last_frame_classed = last_frame.loc[last_frame['class'] == class_id]
    current_frame_classed = current_frame.loc[current_frame['class'] == class_id]
    for i, curr_row in current_frame_classed.iterrows():
      for i, past_row in last_frame_classed.iterrows():
        iou = calc_i_o_u(curr_row['bbox'], past_row['bbox'])
        hist = cv.compareHist(curr_row['roi_hist'], past_row['roi_hist'], cv.HISTCMP_CORREL)
        if iou >= iou_threshold and hist >= hist_threshold:
          print('Associated with an item. class id:', class_id)
          past_row_id = past_row['obj_id']
          curr_row['obj_id'] = past_row_id
      result_frame = result_frame.append(curr_row)
  return result_frame

def track(past_frames, current_frame):
  if len(past_frames) == 0: # if no detections in previous frames, don't do any tracking
    return current_frame
  latest_frame_count = past_frames.tail(1)['frame_count'].as_matrix()
  last_frame = past_frames.loc[past_frames['frame_count'] == latest_frame_count[0]]
  if len(last_frame) != 0:
    current_frame = get_matching_objects(last_frame, current_frame, 0.9, 0.9)
  return current_frame
