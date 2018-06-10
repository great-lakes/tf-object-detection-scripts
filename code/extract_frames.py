"""Extract frames from video for labeling
Use openCV to read in video file dataset and output .JPG frames which will enable labeling in future step
Saved frames will be in /dataset/{dataset_name}/frames/raw
Equalize Histagram and Sharpen are examples to pre-process the images before saving

Note, must use file structure as defined in README.md
Arguments:
  [dataset_name]: name of dataset directory inside of datasets/ which contains the video and data. (required)
  [video_file]: name of video inside {dataset_name} including the file format. (required)
Example Usage:
  extract_frames.py [dataset_name] [video_file]
  extract_frames.py custom_baseline rolling_backpack.mp4
"""

import os
import sys
import cv2
import numpy as np
import random
import argparse

# Define Arguments
parser = argparse.ArgumentParser(description='Extract frames from video for labeling')
parser.add_argument('--dataset_name', help='name of dataset directory inside of datasets/ which contains the video and data. (required)', required=True)
parser.add_argument('--video_file', help='name of video inside dataset including the file format. (required)', required=True)

args = parser.parse_args()
argsdict = vars(args)

# Parse Arguments
DATASET_NAME = argsdict['dataset_name']
VIDEO_FILE = argsdict['video_file']
VIDEO_FILE_ALTERED = VIDEO_FILE.replace('.', '_')

# Read video data
VIDEO_PATH = f'../datasets/{DATASET_NAME}/media/{VIDEO_FILE}'
RELATIVE_VIDEO_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + VIDEO_PATH)
vidcap = cv2.VideoCapture(RELATIVE_VIDEO_PATH)

success, image = vidcap.read()
frame_count = 0
success = True

# pattern used to save overhead with randomly generated array
# create array and shuffle values
# used to shuffle frames for random test/eval sets
def createRandArr():
  rand_arr = np.arange(1000)
  random.shuffle(rand_arr)
  return rand_arr

rand_arr = createRandArr()
rand_arr_counter = 0

while success:
  # Equalize Histagram
  # img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
  # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,s0])
  # image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

  # Sharpen
  # kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
  # image = cv2.filter2D(image, -1, kernel)

  # check if array is exhausted
  if len(rand_arr) == frame_count:
    # reset rand_arr
    rand_arr = createRandArr()
    frame_count = 0
    rand_arr_counter += 1

  # save path to capture raw frames
  # random number will help split training/eval/test sets that are pre-shuffled.
  FRAME_PATH = f'../datasets/{DATASET_NAME}/frames/raw/frame{rand_arr[frame_count]:03}_{VIDEO_FILE_ALTERED}_{rand_arr_counter}{frame_count:03}.jpg'
  RELATIVE_FRAME_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + FRAME_PATH)
  print(RELATIVE_FRAME_PATH)
  cv2.imwrite(RELATIVE_FRAME_PATH, image) # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  frame_count += 1
