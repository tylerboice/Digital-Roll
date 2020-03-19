
# Object Detection Pipeline

Welcome to Digital Roll's object detection Pipeline
![example](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/example.jpg)
1. [ Description.](#desc)
2. [ Data Collection App ](#dataCollectApp)
3. [TensorFlow Workbench](#tfWorkbench)
4. [Errors](#errors)

<a name="desc"></a>
# 1. Description
This repo will walk though the process of creating a object detection classifier for mobile applications.

<a name="dataCollectApp"></a>
# 2. Getting Started
First you will need to download the [GitHub repository](https://github.com/tylerboice/Digital-Roll).

Once installed you will need data to train the classifier.  If you do not have images you can take and label images using the data collection app. If you already have the images then you can label them using the labeling application with the GitHub repository. It is recommened that you have 1000 images per classifier for a well trained model
## Data Collection App
TO DO

## Labelling application
The labeling application is from already in the repository. To run the workbench you will need to have a conda environment created. [Create an environment](#env) to install all the packages needed for the enviorment. Once in the environment you cd to the "labelImg" folder located in the root of the repository. Then run:
```bash
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
```
open the folder with all the images you have to label

###  Hotkeys
~~~
+------------+--------------------------------------------+
| Ctrl + u   | Load all of the images from a directory    |
+------------+--------------------------------------------+
| Ctrl + r   | Change the default annotation target dir   |
+------------+--------------------------------------------+
| Ctrl + s   | Save                                       |
+------------+--------------------------------------------+
| Ctrl + d   | Copy the current label and rect box        |
+------------+--------------------------------------------+
| Space      | Flag the current image as verified         |
+------------+--------------------------------------------+
| w          | Create a rect box                          |
+------------+--------------------------------------------+
| d          | Next image                                 |
+------------+--------------------------------------------+
| a          | Previous image                             |
+------------+--------------------------------------------+
| del        | Delete the selected rect box               |
+------------+--------------------------------------------+
| Ctrl++     | Zoom in                                    |
+------------+--------------------------------------------+
| Ctrl--     | Zoom out                                   |
+------------+--------------------------------------------+
| ↑→↓←       | Keyboard arrows to move selected rect box  |
+------------+--------------------------------------------+
~~~
<a name="tfWorkbench"></a>

# 3. TensorFlow Workbench
This is a modified version of the repo https://github.com/zzh8829/yolov3-tf2. This repo is the basis for a workbench that will aid users in training and validating their own data and conversting the models produced into Apple CoreML to be used on apple mobile devices. It is highly recommend that you use a GPU. Since you are training a model from scratch, the more images and classifiers you have, the longer the training process takes. Make sure your system has a [valid GPU](https://developer.nvidia.com/cuda-gpus) before proceeding

## Programs needed (GPU Only)

[CUDA 10.1](https://developer.nvidia.com/cuda-toolkit-archive) - The drivers needed to run TensorFlow GPU. Currently 10.1 is the most stable version that supports TensorFlow 2.0. Use the exe (local) installer type

[CUDNN 7.6.5](https://developer.nvidia.com/rdp/cudnn-archive) - The additional files needed for TensorFlow GPU. You will need to create a NVIDIA Developer account. CUDNN 7.6.5 is the most stable version for CUDA 10.1.

[Visual Studio 2019 (Community)](https://visualstudio.microsoft.com/downloads/) - Needed for TensorFlow GPU with a windows machine

## Installation (GPU Only):

1) Install [CUDA 10.1](https://developer.nvidia.com/cuda-toolkit-archive)and run the CUDA installer

	  a.  Click OK on the first setup prompt
      b.  Click Agree and Continue when prompted about the license agreement
      c.   Select the Express installation
      d. Complete final installation steps

2) Download [CUDNN 7.6.5](https://developer.nvidia.com/rdp/cudnn-archive) and add the CUDNN files to CUDA
      a.  Open the CUDA path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
      b.  Open the CUDNN installer (you may need [7-zip](https://www.7-zip.org/download.html) to unzip the file)      When open it will contain 1 folder.
     c.  Drag all the contents from the cuda folder from step one into the CUDNN folder.
 ![cudnn](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/cudnn.png)

3) Install [Visual Studio 2019 (Community)](https://visualstudio.microsoft.com/downloads/)
     a.

4) Set up environment variables:
     a. Click on start and type “environment variables”. Select “Edit the system environment variables"
     ![environment variables](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/env_var.png)
     b. Click on "environment variables"
     ![system properties](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/system_prop.png)
     c. In the system path section, double click on path:
 ![system variables path](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/sys_var_path.png)
     d. Enter variables manually by clicking “new”. Make sure you select “ok” when done or the variables will not save. Close the window and reopen to ensure the changes were made. The order of the variables does not matter.

```bash
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\libnvvp

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\libx64
```
![edit environment variables](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/edit_env_var.png)


## Set up
#### Install Anaconda
1. Download[Anaconda (Python 3.7 version)](https://www.anaconda.com/distribution/#download-section) and run the installer.
2. Select the "Just Me" option, however if you User contains a space, you will need to select "All Users" instead. This is because Anaconda can't have a directory with a space in it.
3. Select where you want Anaconda to be installed and click "next"
4. Don't Select either of the Advanced Options and click "install"
<a name="env"></a>
5. After installation is finished, type anaconda in the search bar and open anaconda as admin
![anaconda](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/anaconda.png)

#### Create Anaconda Environment

cd into the "Tenorflow2.0-Workbench" directory
```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate cpu
pip install -r requirements-cpu.txt

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate gpu
pip install -r requirements-gpu.txt
```

## Usage

### Before you run, make sure you have the following files:

 - [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) placed in the data folder

- All data collected place in the image folder:
	- All images with their .xml files if the labelling application was used
	- All .xml files if the data collection app was used


```bash
# from the Tensorflow2.0-Workbench directory
python run_workbench.py
```
The workbench will take in all the data placed in the image folder, and train a model based on that data. Use the "help" or "h" command to bring up the command list
 COMMANDS
```
    continue or c                  ==> Continue workbench if training was manually stopped
                                       You can also continue from previous checkpoint or session
                                       example: continue ./saved_session/session_1
    display or d                   ==> Displays current settings
                                            example: change batch_size 3
    help or h                      ==> Brings up this help display
    info or i                      ==> Displays information on the workbench values
    modify or m <variable> <value> ==> Modifys the setting variable to a new value
                                            For lists of values use the modify(m) command without arguments
    load or l <path to pref.txt>   ==> Loads a given .txt file as the current preference text
    quit or q                      ==> Exits the Workbench
    run or r                       ==> Starts the process of training and validation
                                        + Saves the model at given output location
                                          and creates a Apple CoreML converted version
    save or s <new .txt path>      ==> Saves the current settings to the path + name given
                                            example: save C:\\Users\\new_pref.txt
                                            if no argument given, saves in current working directory as preferences_<number>
    test or t <path to image>      ==> Tests a given image using the last checkpoint
    tflite or l                    ==> Converts the current model in at the current output into a tflite model
```
VARIABLES
The workbench runs on may variables which you can modify to create the perfect model for you. Use the "info" or "i" command to display the variables

### Benchmark Tests
TO DO
<a name="errors"></a>
# Errors

If issues arise in your conda environment you can remove and re-add the environment:
##### To remove a conda enviornment:
```bash
#example conda remove --name cpu --all -y
conda remove --name <name of env> --all -y
```
