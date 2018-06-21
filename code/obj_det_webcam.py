"""Streaming Video from openCV
Using openCV to stream video from webcam and use a pre-trained model for object detection
Model is downloaded and used against webcam frames
openCV outputs images in a separate windows when running.

Note, must use file structure as defined in README.md
Arguments:
  [webcam_index]: Index of webcam to use. Most computer cameras are set up 0=front; 1=back;. (optional, default=0)
  
Example Usage:
  obj_det_webcam.py [webcam_index]
  obj_det_webcam.py 1
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

# Used to verify if file exists on machine
from pathlib import Path # Download Model

# Define Arguments
parser = argparse.ArgumentParser(description='Streaming webcam video from openCV into pre-trained OD Model')
parser.add_argument('--webcam_index', help='Index of webcam to use. Most computer cameras are set up 0=front; 1=back; (default=0)', type=int, required=False, default=0)
args = parser.parse_args()
argsdict = vars(args)

# Prase Arguments
WEBCAM_INDEX = argsdict['webcam_index']

# capture streaming feed from webcam.
cap = cv2.VideoCapture(WEBCAM_INDEX)


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

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
LABELS_PATH = '../../models/research/object_detection/data/mscoco_label_map.pbtxt'
RELATIVE_LABELS_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + LABELS_PATH)

NUM_CLASSES = len(label_map_util.get_label_map_dict(RELATIVE_LABELS_PATH))

# ## Download Model
# Must run first time to download/extract model.
model_file = Path(PATH_TO_CKPT)
if not model_file.is_file():
  print("Downloading model " + str(datetime.datetime.utcnow()))
  opener = urllib.request.URLopener()
  opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
  tar_file = tarfile.open(MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())

print("Loading model into memory " + str(datetime.datetime.utcnow()))
# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
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

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
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
      
      # Print detected items
      tempQ = []
      for i in range(len(scores[0])):
        if scores[0][i] > 0.50:
          tempQ.append("id: " + str(classes[0][i]) + ", score: " + str(scores[0][i]))
      print(tempQ)

      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)

      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        sess.close()
        cap.release()
        cv2.destroyAllWindows()
        break
