# Install Guide

## Pre Requiresite
* A Windows 10 Pro PC
* Preferably a dedicated NVIDIA GPU
* Administrator access to download and install

## 0. Software Requirement
1. [Cmder (Download the Full Version)](http://cmder.net/)
2. [Visual Studio Code](https://code.visualstudio.com/)
3. [Visual Studio 2015/2017 Visual C++ Build Tool](https://www.visualstudio.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15) (Follow the install process, The total install is about 1.5gigs)

## 1. Python
1. [Download Anaconda Python 3.6](https://www.anaconda.com/download/) and run installation with default settings. _Note the installation path which will be used in steps 2 and 3_
2. Add the installation directory path to your Path (either user or system variables) in the Environment Variable settings in System Properties. For us it was `C:\Users\{yourUserName}\AppData\Local\Continuum\anaconda3`
3. Also add the `.\anaconda3\Scripts` to your Path (either user or system variables) in the Environment Variable settings in System Properties. For us it was `C:\Users\{yourUserName}\AppData\Local\Continuum\anaconda3\Scripts`
4. To test python, run `python` in your cmder, and you should see `Python 3.6.x |Anaconda, Inc. ...` and a REPL prompt. 
5. Type `exit()` to exit the REPL.
6. To test pip, run `pip -V` and you should see `pip 10.0.x ...`
7. If you have `pip 9.0.x` run `python -m pip install -U pip` to update to the latest version.

## 2. CUDA (if you have a compatible NVIDIA GPU)
0. [What GPU do I have?](https://help.sketchup.com/en/article/36253)
1. [Go to CUDA 9.0 Download](https://developer.nvidia.com/cuda-90-download-archive) - **Follow the link to download CUDA 9.0**, the latest version of CUDA (9.2) is not supported for Tensorflow yet. [See GH Issue](https://github.com/tensorflow/tensorflow/issues/18906)
2. Pick the right configuration for your machine
3. Download the Base Installer (Express option is ok)
4. Download Patch 2 (Released Mar 5, 2018) (Express option is ok)
5. [Download cuDNN for CUDA 9.0](https://developer.nvidia.com/rdp/cudnn-download) _Note: You will need to create a membership account for download access_
6. [Follow the instruction here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#download-windows) _Note: Complete 4.2 to 4.3, skip 4.3.5 (adding to VS Project)_

## 3. TensorFlow
1. With Python, and CUDA/cuDNN installed, run `pip install --upgrade tensorflow-gpu` to install the GPU Version of Tensorflow.
2. Check that your machine has successfully installed tensorflow. In cmder run `python` to open the python repl.
3. Inside the python repl run `import tensorflow` (you may see a float error, this is expected)
4. Run `print(tensorflow.__version__)` and you should see `1.8.0`
5. Run `exit()` to exit the python repl

tensorflow-gpu is now downloaded and installed successfully. _Note: The following output during installation is okay_

```
notebook 5.4.0 requires ipykernel, which is not installed.
jupyter 1.0.0 requires ipykernel, which is not installed.
jupyter-console 5.2.0 requires ipykernel, which is not installed.
ipywidgets 7.1.1 requires ipykernel>=4.5.1, which is not installed.
```

## 4. Install OpenCV for Python
1. Run `pip install --upgrade opencv-contrib-python`
2. Test the installation by running the python REPL (`python` >>> `import cv2` >>> `print(cv2.__version__)`).  You should see `3.4.1`

## 5. Install Object Detection API for Tensorflow
The Object Detection API provides established models optimized for image/video analysis.
Developers can modify and train these models to fit their specific needs.

1. Anaconda should come with these packages but verify they are installed. To do this run:
      ```
      pip install Cython
      pip install pillow
      pip install lxml
      pip install jupyter
      pip install matplotlib
      ```
2. Create a project folder (i.e `C:\Users\{YourUserName}\projects\obj-det-proj`)
3. Go to https://github.com/tensorflow/models and clone this repository (~500mb) into your project folder (i.e `C:\Users\{YourUserName}\projects\obj-det-proj\models`)

### ProtoBuf
1. [Download the protobuf win-zip 3.4.0 file here](https://github.com/google/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip) and extract it in a `protoc` folder in your project folder (i.e `C:\Users\{YourUserName}\projects\obj-det-proj\protoc`)
2. Go back to `{project}/models/research`, and run `"C:\Users\{YourUserName}\projects\obj-det-proj\protoc\bin\protoc" object_detection/protos/*.proto --python_out=.`

### COCO API Install
1. Go back to your project folder (ie `C:\Users\{YourUserName}\projects\obj-det-proj`), we'll denote this path as `{project}` in this doc going forward.
2. Clone the Coco API repo by running `git clone https://github.com/cocodataset/cocoapi.git`
3. Go into `{project}/cocoapi/PythonAPI`, and open up `setup.py` in VS Code.
4. Remove the arguments in the `extra_compile_args` array (line 12), so the line should look like `extra_compile_args=[],` Save, and exit.
5. In the `{project}/cocoapi/PythonAPI` directory, run `python setup.py build_ext --inplace`
6. You should see `Finished generating code /n copying build\lib.win-amd64-3.6\pycocotools\_mask.cp36-win_amd64.pyd -> pycocotools`
7. In the `{project}/cocoapi/PythonAPI` directory, run `rm -rf build` to remove the build directory
8. In the `{project}/cocoapi/PythonAPI` directory, Run `cp -r pycocotools ../../models/research/`.
> Note: folder `cocoapi` and folder `models` should both be in the same directory

### Add PYTHONPATH environement variable
1. Go to environement variable settings and create a new variable named `PYTHONPATH`  with three values:
  * the path of your python.exe
  * the path of your `{project}/models/research`
  * path of `{project}/models/research/slim`.
  
  Concate the three paths with semi-colons **without spaces in between**.

  For example ours looks like: 
  ```
  C:\Users\{YourUserName}\AppData\Local\Continuum\anaconda3;
  C:\Users\{YourUserName}\projects\obj-det-proj\models\research;
  C:\Users\{YourUserName}\projects\obj-det-proj\models\research\slim
  ```
Restart your cmder (exit and open cmder back up), for the environment variable to take effect.
> running `echo %PYTHONPATH%` should print something similar:
`C:\Users\{YourUserName}\AppData\Local\Continuum\anaconda3;C:\Users\{YourUserName}\projects\obj-det-proj\models\research;C:\Users\{YourUserName}\projects\obj-det-proj\models\research\slim`

### Test Object Detection API
1. Go to `{project}/models/research`
2. Run `python object_detection/builders/model_builder_test.py`
3. You should see
```
...............
----------------------------------------------------------------------
Ran 15 tests in 0.184s

OK
```
> Note: If you see some error with `FutureWarning` and `float` types, it is ok.
