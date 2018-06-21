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
TODO:
- software required to run scripts
- upload and reference installation.md

___
## Workflow
This section outlines and talks through the main workflow process of training a TensorFlow Object Detection model using this repository. Each script referenced below can be found in the `/code` directory. Please use the `--help` flag for each Python script to learn more about how to run the scripts. i.e. `python code/{script}.py --help`. Also, each Python script has detailed documentation at the beginning of each file.

> Note: For the remainder of this documentation, the `/tf-object-detection-scripts/datasets/{dataset_name}/` is denoted as `[DS]` for brevity.  
For example: `/tf-object-detection-scripts/datasets/{dataset_name}/media` directory is written as `[DS]/media`

### 1. Extract Frames
Depending on what image/video labeling tool you use, you may first have to extract the frames from a video that you wish to use. If you have a directory of images you can skip this step. Utilizing the `extract_frames.py` script requires that your video file is located inside the `[DS]/media` directory. 

```
python code/extract_frames.py --dateset_name=example_dataset --video_file=example.mp4
```

Running this script will extract the frames as individual `.jpg` files into `[DS]/frames/raw`

### 2. Label
This is arguably the most important and tedious step when creating a custom object detection model. We are required to label each frame with the objects of interest to our model. There are many open source tools available that you can use for this process, we found [labelImg](https://github.com/tzutalin/labelImg) to be efficient.

The key output that we require from this step is a directory of `.xml` files which contain information about the frame, labels and coordinates. Most labeling tools will allow you to export these files.

Save exported `.xml` files in `[DS]/frames/labeled`

### 3. Convert .xml to .csv
- `/code/convert_xml_to_csv.py`

### 4. Convert .csv to .record
- `/code/convert_csv_to_record.py`

### 5. Train
- `../model/reserach/object_detection/train.py`

### 6. Evaluate
- `../model/reserach/object_detection/eval.py`

### 7. Export Frozen Model
- `../model/reserach/object_detection/export_inference_graph.py`

### 8. Run Object Detection Using Custom Trained Model
- `/code/obj_det_custom.py`

___
## .gitignore
Ignoring the following file formats by default to enforce privacy and avoid accidental uploading of sensitive files. Feel free to change this file in root.
```
*.jpg
*.png
*.mp4
*.MOV
*.xml
*.csv
```