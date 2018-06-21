# TensorFlow + Object Detection Scripts

## Purpose
This repository is meant to help organize and streamline the workflow when creating custom object detection models using tensorflow.

Please use the provided file structure for convenience. The python scripts in this repository are dependent on the structure.
___
## File Structure
> Note: `git clone` this repo in the same level directory as your [TensorFlow Models](https://github.com/tensorflow/models) as illustrated below.

```
/tf-object-detection-scripts
  /code
    convert_csv_to_record.py
    convert_xml_to_csv.py
    extract_frames.py
    obj_det_coco.py
    obj_det_custom.py
    obj_det_custom_track.py
    obj_set_webcam.py
    tracker.py
  /datasets
    /example_dataset
      /data
        train.record
        eval.record
        pascal_label_map.pbtxt
        example.config
      /eval
      /export
        frozen_inference_graph.pb
      /frames
        /eval
        /labeled
          example.xml
        /processed
        /raw
          example.jpg
      /media
        example.mp4
      /training
        model.ckpt
  /pretrained
    /ssd_mobilenet_v1_coco_checkpoint
      frozen_inference_graph.pb
      model.ckpt
/models
  /research
    /object_detection
```
Above shows the file format used as well as where the files are used.

___
## Prerequisites
You will need tensorflow and python 3.6 to run the scripts in this repo.

Please refer to the [INSTALL.md](INSTALL.md) documentation for details on how to install all the dependencies.

___
# Workflow
This section outlines and talks through the main workflow process of training a TensorFlow Object Detection model using this repository. 

**All scripts should be executed directly under the `tf-object-detection-scripts/` directory.**

Each script can be found in the `code/` directory. Please use the `--help` flag for each Python script to learn more about how to run the scripts. i.e. `python code/{script}.py --help`. Also, each Python script has detailed documentation at the beginning of each file.

> Note: For the remainder of this documentation, the `tf-object-detection-scripts/datasets/{dataset_name}/` path is denoted as `[DS]` for brevity.  
For example: `tf-object-detection-scripts/datasets/{dataset_name}/media/` directory is written as `[DS]/media/`

## 1. Extract Frames
Depending on what image/video labeling tool you use, you may first have to extract the frames from a video that you wish to use. If you have a directory of images you can skip this step. Utilizing the `extract_frames.py` script requires that your video file is located inside the `[DS]/media/` directory. 

```
python code/extract_frames.py --dateset_name=example_dataset --video_file=example.mp4
```

Running this script will extract the frames as individual `.jpg` files into `[DS]/frames/raw/`

## 2. Label
This is arguably the most important and tedious step when creating a custom object detection model. We are required to label each frame with the objects of interest to our model. There are many open source tools available that you can use for this process, we found [labelImg](https://github.com/tzutalin/labelImg) to be efficient.

The key output that we require from this step is a directory of `.xml` files which contain information about the frame, labels and coordinates. Most labeling tools will allow you to export these files.

Save exported `.xml` files in `[DS]/frames/labeled/`

## 3. Create .pbtxt file
A label `.pbtxt` file is used by Tensorflow to associate ids to its label text.

Each label should be an `item` in the `.pbtxt` file.  And the value of the `name` property should be the label text.

> Note: Ids should start from 1.

Here is an example of a `.pbtxt` file that consists of 4 labels.

```txt
item {
 id: 1
 name: 'person'
}
item {
 id: 2
 name: 'suitcase'
}
item {
 id: 3
 name: 'backpack'
}
item {
 id: 4
 name: 'handbag'
}
```

Save this file in `[DS]/data/pascal_label_map.pbtxt`

## 4. Convert .xml to .record
TensorFlow requires labeled data to be converted into `.record` files for training. We have provided a script that converts `.xml` files into `.csv`, and ultimately `.record` files.

Example running script:
```
python code/convert_xml_to_record.py --dataset_name=example_dataset --pbtxt_file=pascal_label_map.pbtxt
```
Breaking this conversion into two main steps (already taken care of by the script):

### Step 1: `xml` -> `csv`
The `.xml` files containing the frame name, labels and coordinates will be converted into `.csv` files. This is a convenience step for the developer, allowing easy viewing and manipulation, if needed through programs such as Excel. The resulting `.csv` files will be used to create the `.record` files in the next step of this script.


Example output from the first step of the script (using default 80% train/eval split):
```
Generating .csv files for .record creation...
5082 labels total
4061 labels in train
1021 labels in eval
Successfully converted xml to csv.
```
It is important to be consistent when labeling frames. If there are any labels that are not contained in `.pbtxt` file, you will see an error:
```
ERROR: Value(s) in csv is not in pbtxt. Ensure correct labels are in csv and rerun script.
```
Simply fix the labels inside the `.csv` files and rerun the command.

### Step 2: `csv` -> `record`
At this point, either step 1 has successfully created `.csv` files, or there are detected `.csv` files already inside `[DS]/data/`. If there are any `.csv` file detected, you will see this prompt:
```
.csv files found, skipping .xml to .csv creation.
```

Regardless, the script will use the existing `.csv` files to create TensorFlow's `.record` files which will enable training of a model in the next step.

Example output from the second step of the script:
```
-----------------------
Successfully created the TFRecords: [DS]/data/train.record
Successfully created the TFRecords: [DS]/data/eval.record
-----------------------
Deleting .csv files, please use --keep_csv flag to prevent deletion.
```

> Note the .csv files are deleted in the example above, use the `--keep_csv` flag to keep the created `.csv` files inside `[DS]/data/`

## 5. Download Pretrained Model
TODO:

Get model from model zoo and put into `pretrained/` directory. we require the `.ckpt` files

## 6. Create .config File
TODO:

create example.config

## 7. Train
The training script, provided by TensorFlow, will be used to train our custom model.

Example running script:
```
python ../model/reserach/object_detection/train.py --logtostderr --train_dir=[DS]/training --pipeline_config_path=[DS]/example.config
```

## 8. Evaluate
- `../model/reserach/object_detection/eval.py`

## 9. Export Frozen Model
- `../model/reserach/object_detection/export_inference_graph.py`

## 10. Run Object Detection Using Custom Trained Model
- `/code/obj_det_custom.py`

___
## .gitignore
Ignoring the following file formats by default to enforce privacy and avoid accidental uploading of sensitive files. Feel free to change the `.gitignore` which is located in root.
```
*.jpg
*.png
*.mp4
*.MOV
*.xml
*.csv
```