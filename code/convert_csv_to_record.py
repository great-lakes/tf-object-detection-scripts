"""Convert .csv files to record files used for training
Will generate a train.record and eval.record which tensorflow train.py requires.
train_labels.csv -> train.record
eval_labels.csv -> eval.record

Note, must use file structure as defined in README.md
Arguments:
  [dataset_name]: name of dataset directory inside of datasets/ which contains the .csv data. (required)
  [pbtxt_file]: name of file which defines label indexes including the file format. located in /dataset/{dataset_name}/data directory. (required)
  [num_classes]: number of classes used to train object detection model. (required)
Example Usage:
  convert_csv_to_recrod.py [dataset_name] [pbtxt_file] [num_classes]
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import io
import pandas as pd
import tensorflow as tf
import argparse

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# Use sys to append relative path for libraries to import
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../../models/research/object_detection'))
 
from utils import label_map_util
from utils import visualization_utils as vis_util


# Define Arguments
parser = argparse.ArgumentParser(description='Convert .csv files to record files used for training')
parser.add_argument('--dataset_name', help='name of dataset directory inside of datasets/ which contains the video and data. (required)', required=True)
parser.add_argument('--pbtxt_file', help='name of file which defines label indexes including the file format. located in /dataset/{dataset_name}/data directory. (required)', required=True)
parser.add_argument('--num_classes', help='number of classes used to train object detection model. (required)', type=int, required=True)

args = parser.parse_args()
argsdict = vars(args)

# Parse Arguments
DATASET_NAME = argsdict['dataset_name']
PBTXT_FILE = argsdict['pbtxt_file']
NUM_CLASSES = argsdict['num_classes']

# ## Loading label map
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = f'../datasets/{DATASET_NAME}/data/{PBTXT_FILE}'
RELATIVE_LABELS_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + PATH_TO_LABELS)

# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(RELATIVE_LABELS_PATH)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Filter through .pbtxt to return integer id
def class_text_to_int(row_label):
    indexList = list(filter((lambda category: category['name'] == row_label), categories))

    # check if list is empty before indexing
    if not indexList:
        sys.exit("Value in csv is not in pbtxt. Ensure labels in csv are correct.")

    index = indexList[0]['id']

    return index


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def convert_to_record(name):
    OUTPUT_PATH = f'../datasets/{DATASET_NAME}/data/{name}.record'
    RELATIVE_OUTPUT_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + OUTPUT_PATH)

    writer = tf.python_io.TFRecordWriter(RELATIVE_OUTPUT_PATH)

    CSV_PATH = f'../datasets/{DATASET_NAME}/data/{name}_labels.csv'
    RELATIVE_CSV_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + CSV_PATH)

    FRAME_PATH = f'../datasets/{DATASET_NAME}/frames/raw'
    RELATIVE_FRAME_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + FRAME_PATH)

    examples = pd.read_csv(RELATIVE_CSV_PATH)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, RELATIVE_FRAME_PATH)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = RELATIVE_OUTPUT_PATH
    print('Successfully created the TFRecords: {}'.format(RELATIVE_OUTPUT_PATH))

def main(_):
    convert_to_record('train')
    convert_to_record('eval')

if __name__ == '__main__':
    tf.app.run()