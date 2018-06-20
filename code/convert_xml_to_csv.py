"""Convert labeled .xml to .csv
Parses all .xml files in datasets/{dataset_name}/frames/raw and generates three .csv files
training, eval and combined labels.

Note, must use file structure as defined in README.md
Arguments:
  [dataset_name]: name of dataset directory inside of datasets/ which contains the .xml data. (required)
  [split_value]: value used to split training and eval datasets between 51-99 (default=80)
Example Usage:
  convert_xml_to_csv.py [dataset_name] [split_value]
"""

import os
import sys
import math
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import argparse

# Define Arguments
parser = argparse.ArgumentParser(description='Convert labeled .xml to .csv')
parser.add_argument('--dataset_name', help='name of dataset directory inside of datasets/ which contains the .xml data. (required)', required=True)
parser.add_argument('--split_value', help='value used to split training and eval datasets between 51-99 (default=80)', type=int, required=False, default=80)

args = parser.parse_args()
argsdict = vars(args)

# Parse Arguments
DATASET_NAME = argsdict['dataset_name']

SPLIT_VALUE = argsdict['split_value'] / 100
if not 1 > SPLIT_VALUE > 0.50:
    sys.exit('split_value must be between 51-99')

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

def main():
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

main()
