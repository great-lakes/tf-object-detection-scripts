"""Run object detection on video with custom model with Tracking
Use openCV to read video and apply custom object detection model onto frames
Output frames will be shown on screen as well as saved to frames/eval directory

Note, must use file structure as defined in README.md
Arguments:
  [dataset_name]: name of dataset directory inside of datasets/ which contains the video and data. (required)
  [video_file]: name of video inside dataset including the file format. (required)
  [pbtxt_file]: name of file which defines label indexes including the file format. located in /dataset/{dataset_name}/data directory. (required)
  [num_classes]: number of classes used to train object detection model. (required)
Example Usage:
  obj_det_video_eval_custom.py [dataset_name] [video_file] [pbtxt_file] [num_classes]
  obj_det_video_eval_custom.py custom_baseline rolling1.mp4 pascal_label_map.pbtxt 4
"""

import os
import sys
import cv2
import numpy as np
import datetime # Time stamping
import six.moves.urllib as urllib # Download Model
import tarfile
import tensorflow as tf
import argparse
import tracker

# Used to verify if file exists on machine
from pathlib import Path # Download Model

# Define Arguments
parser = argparse.ArgumentParser(description='Run object detection on video with custom model and tracking')
parser.add_argument('--dataset_name', help='name of dataset directory inside of datasets/ which contains the video and data. (required)', required=True)
parser.add_argument('--video_file', help='name of video inside dataset including the file format. (required)', required=True)
parser.add_argument('--pbtxt_file', help='name of file which defines label indexes including the file format. located in /dataset/{dataset_name}/data directory. (required)', required=True)

args = parser.parse_args()
argsdict = vars(args)

# Parse Arguments
DATASET_NAME = argsdict['dataset_name']
VIDEO_FILE = argsdict['video_file']
VIDEO_NAME_ALTERED = VIDEO_FILE.replace('.', '_')
PBTXT_FILE = argsdict['pbtxt_file']

# ## Object detection imports
# Here are the imports from the object detection module.
# Use sys to append relative path for libraries to import
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../../models/research/object_detection'))
 
from utils import label_map_util
from utils import visualization_utils as vis_util

# Disables the AVX2 warning
# Warning means CPU can utilize advanced vextor extensions on x86 arch for faster results
# Disable warning because we are using GPU
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Model preparation 

# ## Variables
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)
# for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# # What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# SET_VALUE of path with exported .pb file
PATH_TO_CKPT = f'../datasets/{DATASET_NAME}/export/frozen_inference_graph.pb'
RELATIVE_CKPT_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + PATH_TO_CKPT)

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = f'../datasets/{DATASET_NAME}/data/{PBTXT_FILE}'
RELATIVE_LABELS_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + PATH_TO_LABELS)

NUM_CLASSES = len(label_map_util.get_label_map_dict(RELATIVE_LABELS_PATH))

print("Loading model into memory " + str(datetime.datetime.utcnow()))
# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(RELATIVE_CKPT_PATH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

print ("Loading label map " + str(datetime.datetime.utcnow()))
# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(RELATIVE_LABELS_PATH)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print("Detection " + str(datetime.datetime.utcnow()))
# # Detection

EVAL_PATH = f'../datasets/{DATASET_NAME}/eval'
RELATIVE_EVAL_PATH=os.path.abspath(os.path.dirname(__file__) + '/' + EVAL_PATH)

# Read video data
VIDEO_PATH = f'../datasets/{DATASET_NAME}/media/{VIDEO_FILE}'
RELATIVE_VIDEO_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + VIDEO_PATH)
vidcap = cv2.VideoCapture(RELATIVE_VIDEO_PATH)

success, image_np = vidcap.read()
frame_count = 0
past_frames = []

def cleanup():
  sess.close()
  vidcap.release()
  cv2.destroyAllWindows()
  hist_removed = past_frames.drop(['roi_hist'], axis=1)
  hist_removed.to_csv(RELATIVE_EVAL_PATH + '/frames.csv', index=None)
  print('NOTE: a frames.csv file has been generated under the eval directory')

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while success:
      # rotate image
      # (h, w) = image_np.shape[:2]
      # center = (w / 2, h / 2)
      # M = cv2.getRotationMatrix2D(center, -5, 1.0)
      # image_np = cv2.warpAffine(image_np, M, (w, h))

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
      
      # Process Outputs
      squeezed_boxes = np.squeeze(boxes)
      squeezed_classes = np.squeeze(classes).astype(np.int32)
      squeezed_scores = np.squeeze(scores)

      # Process Frames (generate Object ID) and Track Frames
      print('======= Frame ' + str(frame_count) + ' =========')
      current_frame = tracker.make_frame_info(frame_count, squeezed_boxes, squeezed_classes, squeezed_scores, image_np.shape, image_np)
      current_frame = tracker.track(past_frames, current_frame)
      past_frames = tracker.append_current_to_past_frame_info(past_frames, current_frame)
      print('(---- End of Frame ' + str(frame_count) + ' ----) \n')

      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        squeezed_boxes,
        squeezed_classes,
        squeezed_scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)

      # save path to capture raw frames
      FRAME_PATH = f'../datasets/{DATASET_NAME}/frames/eval/frame_{VIDEO_NAME_ALTERED}_{frame_count}.jpg'
      RELATIVE_FRAME_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + FRAME_PATH)
      cv2.imwrite(RELATIVE_FRAME_PATH, image_np) # save frame as JPEG file

      cv2.imshow('object detection', image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cleanup()
        break

      success, image_np = vidcap.read()
      frame_count += 1
      if not success:
        cleanup()
        break
        