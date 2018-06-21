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
You will need TensorFlow and Python 3.6 to run the scripts in this repo.

Please refer to the [INSTALL.md](INSTALL.md) documentation for details on how to install all the dependencies.
> Make sure you have done step '5. Install Object Detection API for Tensorflow' including running the `protoc` command.
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
python code/extract_frames.py \
    --dateset_name=example_dataset \
    --video_file=example.mp4
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
python code/convert_xml_to_record.py \
    --dataset_name=example_dataset \
    --pbtxt_file=pascal_label_map.pbtxt
```
Breaking this conversion into two main steps (already taken care of by the script):

**Step 1: `xml` -> `csv`**

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

**Step 2: `csv` -> `record`**

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

1. Download a pretrained model from Tensorflow's [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).  
2. Extract the tar file as a directory under `tf-object-detection-scripts/pretrained/`.
    * For example, it should looks like `tf-object-detection-scripts/pretrained/ssd_mobilenet_v1_coco_checkpoint/`
3. In this directory, there should be three files with names starting with `model.ckpt...`

## 6. Create a `.config` File
A pipeline configuration file is needed for Tensorflow to know how to train a model.

A pipeline consists of multiple aspects of the training process.  Here are some aspects of the pipeline config file:
* The model being trained, along with layer configurations
* Hyperparameters
* Pretrained checkpoint location for Transfer Learning
* Paths for important files like `.record` and label `.pbtxt` files.

To create a `.config` file, the best way is to copy one from `models/research/object_detection/samples/configs/`, and pick the model that you want to apply transfer learning on.  Copy the selected config file under `[DS]/data/{model_name}.config`

**Changing the copied `.config` file:**
* Search for `batch_size` and lower the value to `2`.  If you are using a dedicated VM for deep learning, you can increase this number.  Larger the number, less fluctuation of the loss value.
* Search for `fine_tune_checkpoint` and set the value to `./models/{dir_from_model_zoo}/model.ckpt`
* Search for `train_input_reader` and set the value of `tf_train_input_reader -> input_path` to `./datasets/{dataset_name}/data/train.record`.  And set `label_map_path` to `./datasets/{dataset_name}/data/pascal_label_map.pbtxt`
* Search for `eval_input_reader` and set the value of `tf_train_input_reader -> input_path` to `./datasets/{dataset_name}/data/train.record`.  And set `label_map_path` to `./datasets/{dataset_name}/data/pascal_label_map.pbtxt`

Below is an example of all the values being changed.
```
...
train_config: {
  batch_size: 2
  ...
  fine_tune_checkpoint: "./pretrained/{dir_from_model_zoo}/model.ckpt"
  from_detection_checkpoint: true
  ...
}

...

train_input_reader: {
  tf_record_input_reader {
    input_path: "./datasets/{dataset_name}/data/train.record"
  }
  label_map_path: "./datasets/{dataset_name}/data/pascal_label_map.pbtxt"
}

...

eval_input_reader: {
  tf_record_input_reader {
    input_path: "./datasets/{dataset_name}/data/eval.record"
  }
  label_map_path: "./datasets/{dataset_name}/data/pascal_label_map.pbtxt"
  shuffle: false
  num_readers: 1
}
```

## 7. Train
The training script, provided by TensorFlow, will be used to train our custom model. Please refer to the TensorFlow documentation per file for more information. The two key arguments we must pass is the path to a training directory for the training checkpoint files to output. We recommend using the `[DS]/training` directory provided as a central place to store training files. The second is the `.config` file created in the previous step.

> Note: You can stop the training at any time with `ctrl+c` and the checkpoint files will remain in the training directory.

Example running script:
```
python ../models/research/object_detection/train.py \
    --logtostderr \
    --train_dir=datasets/{dataset_name}/training \
    --pipeline_config_path=datasets/{dataset_name}/data/example.config
```
> Note: The argument paths are relative to the location the command is being executed.

## 8. Export Frozen Model
Next, we require a frozen model to consume in the format of a `.pb` file. We will again utilize a script provided by TensorFlow to export a model from the resulting `.ckpt` checkpoint files inside of `[DS]/training` from the previous step. Notice that the training resulted in many `model.ckpt` files with numbers that indicate which step the checkpoint was trained until. i.e. `model.ckpt-2300`. Pass in the desired checkpoint model prefix when executing the script.

Example running script:
```
python ../models/research/object_detection/export_inference_graph.py
    --input_type=image_tensor \
    --pipeline_config_path=datasets/{dataset_name}/data/example.config \
    --trained_checkpoint_prefix=datasets/{dataset_name}/training/model.ckpt \
    --output_directory=datasets/{dataset_name}/export
```

The resulting file will be a `frozen_inference_graph.pb` file inside `[DS]/export`

## 9. Run Object Detection Using Custom Trained Model
The final step is to use the exported frozen model to analyze a video. Similar to the `obj_det_coco.py` script which analyzes a video using the [COCO model](http://cocodataset.org/#home), we can use `obj_det_custom.py` script to use our own custom trained model.

The script will automatically look for the `[DS]/export/frozen_inference_graph.pb` model to use. It is important the `.pb` is located there.

Example running script:
```
python code/obj_det_custom.py \
    --dataset_name=example_dataset \
    --video_file=example.mp4 \
    --pbtxt_file=pascal_label_map.pbtxt
```

The video should be displayed in a new window with the labeled frames. The frames will also be saved to `[DS]/frames/eval` for future reference.

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