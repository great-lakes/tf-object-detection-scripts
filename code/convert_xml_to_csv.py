"""Convert labeled .xml to .csv
Parses all .xml files in datasets/{dataset_name}/frames/raw and generates three .csv files
training, eval and combined labels.

Note, must use file structure as defined in README.md
Arguments:
  [dataset_name]: name of dataset directory inside of datasets/ which contains the .xml data. (required)
Example Usage:
  convert_xml_to_csv.py [dataset_name]
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import argparse

# Define Arguments
parser = argparse.ArgumentParser(description='Convert labeled .xml to .csv')
parser.add_argument('--dataset_name', help='name of dataset directory inside of datasets/ which contains the .xml data. (required)', required=True)

args = parser.parse_args()
argsdict = vars(args)

# Parse Arguments
DATASET_NAME = argsdict['dataset_name']

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

def split(df, chunk_size):
    indices = index_marks(df.shape[0], chunk_size)
    return np.split(df, indices)

def main():
    LABELED_PATH = f'../datasets/{DATASET_NAME}/frames/labeled'
    RELATIVE_LABELED_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + LABELED_PATH)

    SAVE_PATH = f'../datasets/{DATASET_NAME}/data'
    RELATIVE_SAVE_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + SAVE_PATH)


    xml_df = xml_to_csv(RELATIVE_LABELED_PATH)
    chunks = split(xml_df, round(xml_df.shape[0] * 0.8))
    chunks[0].to_csv(RELATIVE_SAVE_PATH + '/train_labels.csv', index=None)
    chunks[1].to_csv(RELATIVE_SAVE_PATH + '/eval_labels.csv', index=None)
    xml_df.to_csv(RELATIVE_SAVE_PATH + '/combined_labels.csv', index=None)
    print('Successfully converted xml to csv.')

main()
