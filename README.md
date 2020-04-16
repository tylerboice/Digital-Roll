# Object Detection Pipeline

Welcome to [Digital Roll's](https://www.cefns.nau.edu/capstone/projects/CS/2020/Digital-Roll-S20/index.html) object detection Pipeline

![example](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/example.jpg)
1. [ Description.](#desc)
2. [ Data Collection App ](#dataCollectApp)
3. [TensorFlow Workbench](#tfWorkbench)
4. [Errors](#errors)

<a name="desc"></a>
# 1. Description
This repo will walk though the process of creating a object detection classifier for mobile applications. The repo features three main features:
1) A data collection application(ios only) that can take images and classification data for training the model
2) A workbench that trains the model and produces a [TensorFlow](https://www.tensorflow.org/learn) model and a [coreml](https://developer.apple.com/machine-learning/models/) model
3) A application(ios only)  that can display models created from the workbench

Currently the data application app is set to detect polyhedral dice; however, it can be repossessed for any custom classifier. This tutorial will be for the detection of polydral dice of sides 4, 6, 8, 10, 12, and 20. The workbench assumes that you know the basics of directories and have some command line interface experience

<a name="dataCollectApp"></a>
# 2. Data Collection Application
First you will need to download the current [GitHub repository](https://github.com/tylerboice/Digital-Roll).

Once installed you will need data to train the classifier.  If you do not have images, you can take and label images using the data collection app. If you already have the images then you can label them using the labeling application with the GitHub repository. It is recommended that you have 1000 images per classifier for a well trained model
## Data Collection Application
TO DO

<a name="labelApp"></a>
## Labelling Application
The labeling application is from this [repo](https://github.com/tzutalin/labelImg) and is already in the repository. To run the workbench you will need to have a conda environment created. [Create an environment](#env) to install all the packages needed for the application. Once in the environment you can cd to the "labelImg" folder located in the root of the repository. Then run:
```bash
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
```
**Usage**
Once that command is ran it will open the application. Within the application use the "Open Dir" button located in the left column to open the folder that contains the images you have taken. Label each image by creating a bounding box around. After a you have finished bounding all the classifiers within the image save the file. When you save the file it will save as a sperate *.xml* that contains the name of the image it is using. **Ensure you are saving the xml files as the same name as the image. If you change the name of the image after you save, the *.xml* file will not be able to find the image, so make sure the image names are the ones you want before you begin this process.**

**Tips**
This can be a long and tedious process. To increase the speed you can use the following tips:
- modify the file .\labelImg\data\predefined_classes to ensure it has all the classes you are using. This will allow the application to give suggestions so you don't have to enter the name of the classifier each time.
- activate auto save mode(View->Auto ) which will continuously save so you can move on to the next photo without needed to save every time. Warning: this may save to the .\labelling\ path, make sure you move to the correct directory(.\Tensorflow2.0-Workbench\images) when done labeling.

- use the hotkeys. For a more in depth tutorial check out the original

| Hotkey  | Usage           |
| ------------- |:-------------|
| Ctrl + u   | Load all of the images from a directory    |
| Ctrl + r   | Change the default annotation target dir   |
| Ctrl + s   | Save                                       |
| Ctrl + d   | Copy the current label and rect box        |
| Space      | Flag the current image as verified         |
| w          | Create a rect box                          |
| d          | Next image                                 |
| a          | Previous image                             |
| del        | Delete the selected rect box               |
| Ctrl++     | Zoom in                                    |
| Ctrl--     | Zoom out                                   |
| ↑→↓←       | Keyboard arrows to move selected rect box  |

<a name="tfWorkbench"></a>
# 3. TensorFlow Workbench
This is a modified version of the [repo](https://github.com/zzh8829/yolov3-tf2). This repo is the basis for a workbench that will aid users in training and validating their own data and conversting the models produced into Apple CoreML to be used on apple mobile devices. It is highly recommend that you use a GPU. Since you are training a model from scratch, the more images and classifiers you have, the longer the training process takes. Make sure your system has a [valid GPU](https://developer.nvidia.com/cuda-gpus) before proceeding.

## Required (GPU Only)

  - CUDA 10.1 - The drivers needed to run TensorFlow GPU. Currently 10.1 is the most stable version that supports TensorFlow 2.0. Use the exe (local) installer type

 - CUDNN 7.6.4 - The additional files needed for TensorFlow GPU. You will need to create a NVIDIA Developer account. CUDNN 7.6.4 is the most stable version for CUDA 10.1.

- Visual Studio 2019 (Community) - Needed for TensorFlow GPU with a windows machine

- At minimum 50GB of free space on your machine 

Installation guide below

## Installation (GPU Only):

1) Install [CUDA 10.1](https://developer.nvidia.com/cuda-toolkit-archive)and run the CUDA installer

	  a) Click OK on the first setup prompt
		b) Click Agree and Continue when prompted about the license agreement
    c) Select the Express installation
    d) Complete final installation steps

2) Download [CUDNN 7.6.4](https://developer.nvidia.com/rdp/cudnn-archive) and add the CUDNN files to CUDA

    a) Open the CUDA path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
    b) Open the CUDNN installer (you may need [7-zip](https://www.7-zip.org/download.html) to unzip the file). When open it will contain 1 folder.
    c) Drag all the contents from the cuda folder from step one into the CUDNN folder.

 ![cudnn](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/cudnn.png)

3) Install [Visual Studio 2019 (Community)](https://visualstudio.microsoft.com/downloads/)

    a) Run the installer
    b) Select the C++ development workload and finish downloading visual studio
		<br/><br/>
![visual_studio](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/visual_studio.png)
<br/><br/>
4) Set up environment variables:

    a) Click on start and type “environment variables”. Select “Edit the system environment variables"

    ![environment variables](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/env_var.png)
<br/><br/>
    b) Click on "environment variables"

     ![system properties](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/system_prop.png)
<br/><br/>
    c) In the system path section, double click on path:

 ![system variables path](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/sys_var_path.png)
<br/><br/>     
	  d) Enter variables manually by clicking “new”. Make sure you select “ok” when done or the variables will not save. Close the window and reopen to ensure the changes were made. The order of the variables does not matter.

```bash
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\libnvvp

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\libx64
```
![edit environment variables](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/edit_env_var.png)


## Set up

#### Install Anaconda
1. Download[Anaconda (Python 3.7 version)](https://www.anaconda.com/distribution/#download-section) and run the installer.
2. Select the "Just Me" option, however if the file-path contains a space, you will need to select "All Users" instead. This is because Anaconda can't have a directory with a space in it.
3. Select where you want Anaconda to be installed and click "next"
4. Don't Select either of the Advanced Options and click "install"
<a name="env"></a>
5. After installation is finished, type anaconda in the search bar and open anaconda as admin
![anaconda](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/anaconda.png)

#### Create Anaconda Environment

cd into the "Tenorflow2.0-Workbench" directory
#### Tensorflow CPU
```bash
conda env create -f conda-cpu.yml
conda activate cpu
pip install -r requirements-cpu.txt

```
#### Tensorflow GPU
```bash
conda env create -f conda-gpu.yml
conda activate gpu
pip install -r requirements-gpu.txt
```

## Usage
### Before you start the workbench, ensure the following:

- All your images are [labelled](#labelApp)

- All data collected is placed in the ./Tensorflow2.0-Workbench/image folder:
	- All images with their .xml files if the labelling application was used
	- All .xml files if the mobile ios data collection app was used

- You are in the conda environment
  - To ensure this, look at then word before your current working directory. By default it says "(base)", it needs to say "(cpu)" or "(gpu)" depending on which you are using.
  - To enter an environment, run:
  ```bash
  #cpu
  conda activate cpu

  #gpu
  conda activate gpu
  ```

### Start the Workbench:
```bash
# from the Tensorflow2.0-Workbench directory
python run_workbench.py
```
The workbench will take in all the data placed in the image folder, and train a model based on that data. Use the "help" or "h" command to bring up the command list

#### COMMANDS
 *accepts word or letter value in the workbench*
```
continue(c)                  ==> Continue training workbench if training was manually stopped (if no argument given)
                                 You can also continue from previous checkpoint or session
                                      example: continue ./saved_session/session_1

display(d)                   ==> Displays current settings

help(h)                      ==> Brings up this help display

info(i)                      ==> Displays breif information on the workbench values

modify(m) <variable> <value> ==> Modifys the setting variable to a new value
                                      example: m batch_size 10
                                 To view list of values use the modify(m) command without arguments

load(l) <path to pref.txt>   ==> Loads a given .txt file as the current preference text

quit(q)                      ==> Exits the Workbench

run(r)                       ==> Starts the process of training and validation
                                        + Saves the model at given output location
                                          and creates a Apple CoreML converted version

save(s) <new .txt path>      ==> Saves the current settings to the path + name given
                                 If no argument given, saves in current working directory as preferences.txt
                                      example: save C:\Users\new_pref.txt


test(t) <path to image>      ==> Tests a given image using the last checkpoint
									  example: test C:\Users\test_image.jpg
```

#### VARIABLES

*The workbench runs on may variables which you can modify to create the perfect model for you. Use the info(i) command to display the variables*
<pre>
batch_size - Integer
            number of examples that are trained at once. Batch size is depended
            on  hardware. Better hardware can support a higher batch size.

classifiers - String - .names file
            the name of the classifier file you are training from. By default the
            workbench will create a file ./data/classifier.names. The classifier
            file is a .names file that has the name of each classifier on a new line.

dataset_test - String - .tfrecord file
            when tensorflow runs it uses two .tfrecord files, one for testing and
            one for training. This is the .tfrecord file that is used for testing.
            by default the workbench will create a file ./data/test.tfrecord using
            the images and .xml files in the image folder. This variable will most
            likely never need to be modified

dataset_train - String - .tfrecord file
            when tensorflow runs it uses two .tfrecord files, one for testing and
            one for training. This is the .tfrecord file that is used for training.
            by default the workbench will create a file ./data/train.tfrecord using
            the images and .xml files in the image folder. This variable will most
            likely never need to be modified

epochs - Integer
            a epoch is one iteration through all the images and xml files given to the
            workbench. The more epochs the more the workbench iterates through the
            dataset; therefore, the more trained the model will be.

image_size - Integer - 256 or 416
	        This workbench uses the yolo method of training. In summary this method
	        divides the image into boxes to predict where classifiers will be. Image
	        size determines how many boxes are created. While 416 will make more boxes,
	        it will also be slower and require more processing power than 256.

max_checkpoints - Integer
            after every iteration(epoch), the workbench will save a checkpoint. These
            checkpoints take up a lot of memory. This variable is the amount of checkpoints
            that the checkpoint will keep before deleting older checkpoints. Therefore
            if you have 5 max checkpoints then the workbench will never have more than 5
            checkpoints saved. On the 6th epoch, the first checkpoint will be deleted.

max_sessions - Integer
	        once the model is finished all the output from the workbench is saved. If the
	        workbench is ran again, then that output is saved to a saved_session folder.
	        This variable determines the amount of sessions that are saved before older
	        checkpoints are deleted. Therefore if you have 5 max sessions then the
	        workbench  will never have more than 5 sessions saved. On the 6th session, the
	        first session will be deleted.

	        Note: if the name of a saved_session folder is modified,
	              the workbench will not delete that session

mode - String - must be 'fit', 'eager-fit' or 'eager-tf'
            TO DO
                    - fit:
                    - eager-fit:
                    - eager_tf:

output - String - file-path
             folder where all workbench output is placed including:
                    - saved checkpoints
                    - models created
                    - images tested

pref - String - .txt file
             a .txt file that contains all the variables. These can be used to as
             speific preferences for training the workbench, instead of having
             to edit every variable. Use the load(l) command to load in a preference
             file. Use the save(s) command to save the current preferences to a file.

sessions - String - file-path
            once the model is finished all the output from the workbench is saved. If the
	        workbench is ran again, then that output is saved to a saved_session folder.     
	        This variable should be a file-path where you want the sessions saved

tiny_weights - Boolean - True or False
            TO_DO

transfer - String - 'none', 'darknet', no_output', frozen', or 'fine_tune'
            TO_DO
                    - none:
                    - darknet:
                    - no_output:
                    - frozen:
                    - fine_tune:

val_img_num - Integer
             When the workbench is first ran it divided the images and their xml files
             into three folders: test, train, and validate. The validate folder is not used
             in the model. Instead it is used after the model is created to test the accuracy
             of the model. This variable determines how many images should be removed
             from the workbench to test on the model

val_image_path - String - file-path
             When the workbench is first ran it divided the images and their xml files
             into three folders: test, train, and validate. The validate folder is not used
             in the model. Instead it is used after the model is created to test the accuracy
             of the model. This variable is the folder that the validate images are stored

weighted_classes - Integer
              number of classifiers that was used for the weights file. If using the
              default weights or tiny weights file, they are trained on 80 classes

weights - Sting - .weights or .tf file
              when the workbench trains a model, it uses a pre-trained model to assist it
              this variable is the path to the pre-trained model.
</pre>
### Running the workbench
When the workbench runs, the first thing it will do is organize all the data in the images folder. It will then print all the images, the amount of classifiers found, and how many of each classifier were found. Additionally, it will print any images that do not have a corresponding xml file.

After it gathers all the data it will split it into three separate folders: train, test and validate. The train will be 90% of the images, test data will be 10% and the validate will be the *val_img_num* amount of images. The test and train folders will then be generated into .tfrecords.

Then the pre-trained weights file will be converted to a checkpoint for training:
 ![convert_weights](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/convert.PNG)
 <br/> <br/>
After the checkpoint has been converted, the workbench will begin to train the workbench. This will look like this:

![workbench_output](https://raw.githubusercontent.com/tylerboice/Digital-Roll/master/Tensorflow2.0-Workbench/docs/epochs.PNG)
 <br/>
 The most important category to focus on is the loss. If you have a high loss rate, the model is still inaccurate. If your classifiers are very similar or if you are training from scratch, the loss rate will start very high. Ideally you want a loss rate below 1, however a loss rate below 5 is a well trained model.

After the training is done, the workbench it will create a coreML model and a .pb model. It will train this model on the validate images and output how the objects it found within the image. When it detects an object, it is anything it things is a classifier with a +50% accuracy. So if it thinks it can be one of many options it will display all the options.

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
