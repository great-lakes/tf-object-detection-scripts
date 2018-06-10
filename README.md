# TensorFlow + Object Detection Scripts

## Purpose
This repository is meant to help organize and streamline the workflow when creating custom object detection models using tensorflow.

Please use the provided file structure for convenience. The python scripts in this repository are dependant on the structure.
___
## File Structure
> Note: `git clone` this repo in the same level directory as your [TensorFlow Models](https://github.com/tensorflow/models) as illistrated below.

```
/tf-object-detection-scripts
  /code
    convert_csv_to_record.py
    convert_xml_to_csv.py
    extract_frames.py
    obj_det_coco.py
    obj_det_custom.py
    obj_set_webcam.py
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
  /models
      /ssd_mobilenet_v1_coco_checkpoint
          frozen_inference_graph.pb
          model.ckpt
/models
  /research
    /object_detection
```
Above shows the file format used as well as where key files used throughout the workflow are located.

___
## Workflow
### 1. Extract Frames
- `/code/extract_frames.py`

### 2. Label
- We found [labelImg](https://github.com/tzutalin/labelImg) to be efficient.

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