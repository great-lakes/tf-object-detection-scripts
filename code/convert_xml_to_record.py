"""Convert labeled .xml to .record files
Step 1:
Parses all .xml files in datasets/{dataset_name}/frames/raw and generates three .csv files
training, eval and combined labels.

Step 2:
Will generate a train.record and eval.record which tensorflow train.py requires.
train_labels.csv -> train.record
eval_labels.csv -> eval.record

Running this script with csv files detected will start script at step 2.

Note, must use file structure as defined in README.md
Arguments:
  [dataset_name]: name of dataset directory inside of datasets/ which contains the .xml data. (required)
  [pbtxt_file]: name of file which defines label indexes including the file format. located in /dataset/{dataset_name}/data. (required)
  [split_value]: value used to split training and eval datasets between 51-99 (default=80)
  [keep_csv]: boolean value to keep .csv files for developer reference (default=false)

Example Usage:
  convert_xml_to_record.py [dataset_name] [pbtxt_file] [split_value] [keep_csv]
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import sys
import math
import glob
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# Use sys to append relative path for libraries to import
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../../models/research/object_detection'))
 
from utils import label_map_util
from utils import visualization_utils as vis_util

# Define Arguments
parser = argparse.ArgumentParser(description='Convert labeled .xml to .csv')
parser.add_argument('--dataset_name', help='name of dataset directory inside of datasets/ which contains the .xml data. (required)', required=True)
parser.add_argument('--pbtxt_file', help='name of file which defines label indexes including the file format. located in /dataset/{dataset_name}/data. (required)', required=True)
parser.add_argument('--split_value', help='value used to split training and eval datasets between 51-99 (default=80)', type=int, default=80, required=False)
parser.add_argument('--keep_csv', help='include flag to keep .csv files for developer reference (default=false)', dest='keep_csv', action='store_true', required=False)
parser.set_defaults(keep_csv=False)

args = parser.parse_args()
argsdict = vars(args)

# Parse Arguments
DATASET_NAME = argsdict['dataset_name']
PBTXT_FILE = argsdict['pbtxt_file']
KEEP_CSV = argsdict['keep_csv']

SPLIT_VALUE = argsdict['split_value'] / 100
if not 1 > SPLIT_VALUE > 0.50:
    sys.exit('split_value must be between 51-99')

"""
CONVERT XML TO CSV
"""

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

#splits into parts based on chunk_size
def split(df, chunk_size):
    indices = index_marks(df.shape[0], chunk_size)
    return np.split(df, indices)

def convert_xml_to_csv():
    LABELED_PATH = f'../datasets/{DATASET_NAME}/frames/labeled'
    RELATIVE_LABELED_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + LABELED_PATH)

    SAVE_PATH = f'../datasets/{DATASET_NAME}/data'
    RELATIVE_SAVE_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + SAVE_PATH)

    xml_df = xml_to_csv(RELATIVE_LABELED_PATH)

    # separate labeled frames based on defined train/eval split

    total_label_count = xml_df.shape[0]
    cutoff_label_index = math.floor(total_label_count * SPLIT_VALUE)

    # find filename at cutoff index to ensure the labels referencing a single file are not split
    cutoff_label_filename = xml_df.ix[cutoff_label_index]['filename']
    filename_df = xml_df.loc[xml_df['filename'] == cutoff_label_filename].reset_index()
    filename_index = filename_df.ix[0]['index']
    # update new index value that does not split frame labels
    cutoff_label_index = filename_index
    
    chunks = split(xml_df, cutoff_label_index)
    print(f'{total_label_count} labels total')

    # training labels
    chunks[0].to_csv(RELATIVE_SAVE_PATH + '/train_labels.csv', index=None)
    train_label_count = chunks[0].shape[0]
    print(f'{train_label_count} labels in train')

    # eval labels
    chunks[1].to_csv(RELATIVE_SAVE_PATH + '/eval_labels.csv', index=None)
    eval_label_count = chunks[1].shape[0]
    print(f'{eval_label_count} labels in eval')

    # combined labels
    xml_df.to_csv(RELATIVE_SAVE_PATH + '/combined_labels.csv', index=None)

    # successful print
    print('Successfully converted xml to csv.')
    print('-----------------------')

"""
CONVERT CSV TO RECORD
"""
# TODO: call function to dynamically get # of classes. Function should return integer 
NUM_CLASSES=4

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
        sys.exit("ERROR: Value(s) in csv is not in pbtxt. Ensure correct labels are in csv and rerun script.")

    index = indexList[0]['id']

    return index

def split2(df, group):
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
    grouped = split2(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, RELATIVE_FRAME_PATH)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = RELATIVE_OUTPUT_PATH
    print('Successfully created the TFRecords: {}'.format(RELATIVE_OUTPUT_PATH))

def convert_csv_to_record():
    convert_to_record('train')
    convert_to_record('eval')

def remove_file(file_path):
  if os.path.exists(file_path):
    os.remove(file_path)

def remove_csv():
  # check keep_csv flag to determine keep/delete
  if not KEEP_CSV:
    # check if both eval.csv exists to delete
    PATH_TO_EVAL = f'../datasets/{DATASET_NAME}/data/eval_labels.csv'
    remove_file(os.path.abspath(os.path.dirname(__file__) + '/' + PATH_TO_EVAL))

    # check if both train.csv exists to delete
    PATH_TO_TRAIN = f'../datasets/{DATASET_NAME}/data/train_labels.csv'
    remove_file(os.path.abspath(os.path.dirname(__file__) + '/' + PATH_TO_TRAIN))

    # check if both combined.csv exists to delete
    PATH_TO_COMBINED = f'../datasets/{DATASET_NAME}/data/combined_labels.csv'
    remove_file(os.path.abspath(os.path.dirname(__file__) + '/' + PATH_TO_COMBINED))

"""
MAIN FUNCTION
"""

def main():
  convert_xml_to_csv()
  convert_csv_to_record()
  remove_csv()
  
main()