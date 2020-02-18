import os
import re
import sys

from absl import app, flags, logging
from absl.flags import FLAGS
from os import path

CWD_PATH = os.getcwd().replace("\\", "/") + "/"

# IMAGE SET PATHS
IMAGES_PATH = os.path.join(CWD_PATH , "images/")
INPUT_IMAGE_PATH = os.path.join(IMAGES_PATH , "validate/")
TRAIN_IMAGE_PATH = os.path.join(IMAGES_PATH , "train/")
TEST_IMAGE_PATH = os.path.join(IMAGES_PATH , "test/")
VALIDATE_IMAGE_PATH = os.path.join(IMAGES_PATH , "validate/")

# DATA PATHS
DATA_PATH = os.path.join(CWD_PATH , "data/")

CLASSIFIER_FILE = os.path.join(DATA_PATH , "classifier.names")
TRAIN_TF_RECORD_PATH = os.path.join(DATA_PATH , "train.tfrecord")
TEST_TF_RECORD_PATH = os.path.join(DATA_PATH , "test.tfrecord")

# CHECKPOINTS
CHECKPOINT_PATH = os.path.join(CWD_PATH , "checkpoints/")

# OUTPUT
OUTPUT_PATH = os.path.join(CWD_PATH , "output/")

# HARD CODED VALUES
MIN_IMAGES = 50
BOOL = -999
INT = -998
FILE = -997
FLOAT = -996

TRANSFER_OPTIONS = ['none', 'darknet', 'no_output', 'frozen', 'fine_tune']
MODE_OPTIONS = ['fit', 'eager_fit', 'eager_tf']
NO_PREF_PATH = "none"
YOLO_PATH = DATA_PATH + "yolov3.weights"
TINY_PATH = DATA_PATH + "yolov3-tiny.weights"

# Default values
DEFAULT_BATCH_SIZE = 4
DEFAULT_CHECKPOINT_PATH = CWD_PATH + 'checkpoints/yolov3.tf'
DEFAULT_EPOCH_NUM = 10
DEFAULT_HELP = False
DEFAULT_IMAGE_SIZE = 416
DEFAULT_LEARN_RATE = .001
DEFAULT_MODE = 'fit'
DEFAULT_NUM_VAL_IMAGES = 3
DEFAULT_PREF_PATH = CWD_PATH + "preferences.txt"
DFEAULT_TRANSFER_TYPE = 'none'
DEFAULT_WEIGHT_NUM = 80
DEFAULT_WEIGHT_TYPE = False #False = yolov3 True = tiny

# PREFERENCE VARIABLE NAMES
BATCH_SIZE_VAR = "batch_size"
CHECKPOINT_VAR = "checkpoint_path"
CLASSIFIERS_VAR = "classifiers"
DATASET_TEST_VAR = "dataset_test"
DATASET_TRAIN_VAR = "dataset_train"
EPOCH_NUM_VAR = "epochs"
HELP_VAR = "help"
IMAGE_SIZE_VAR = "image_size"
LEARN_RATE_VAR = "learn_rate"
MODE_VAR = "mode"
OUTPUT_VAR = "output"
PREF_VAR = "pref"
TEST_IMAGE_VAR = "validate_images"
TINY_WEIGHTS_VAR = "tiny_weights"
TRANSFER_VAR = "transfer"
VALID_IN_VAR = "validate_image_input"
WEIGHTS_CLASS_VAR = "Weighted_class"
WEIGHTS_PATH_VAR = "weights_path"

CONVERT_WEIGHT = "convert_weight"
CREATE_CLASS_VAR = "create_class_file"
DETECT_IMAGES_VAR = "detect_img"
EXPORT_COREML_VAR = "core_ml"
EXPORT_Tf_MODEL_VAR = "tf_model"
GENRATE_TF_VAR = "generate_tf"
SORT_IMAGES_VAR = "sort_images"
TRAIN_VAR = "train"


########################## FLAGS #################################
########################
###      VALUES      ###
########################
# batch_size
flags.DEFINE_integer( BATCH_SIZE_VAR, DEFAULT_BATCH_SIZE, 'batch size')
# checkpoint path
flags.DEFINE_string( CHECKPOINT_VAR, DEFAULT_CHECKPOINT_PATH, 'path to output')
# classifiers
flags.DEFINE_string( CLASSIFIERS_VAR, CLASSIFIER_FILE, 'file of classifier names')
# dataset_test
flags.DEFINE_string( DATASET_TEST_VAR, TEST_TF_RECORD_PATH, 'path to test tf record')
# dataset_train
flags.DEFINE_string( DATASET_TRAIN_VAR, TRAIN_TF_RECORD_PATH, 'path to training tf record')
# ephoch num
flags.DEFINE_integer( EPOCH_NUM_VAR, DEFAULT_EPOCH_NUM, 'number of epochs')
# help
flags.DEFINE_bool( HELP_VAR, DEFAULT_HELP, "help me")
# image_size
flags.DEFINE_integer( IMAGE_SIZE_VAR, DEFAULT_IMAGE_SIZE, 'image size')
# learn rate
flags.DEFINE_float( LEARN_RATE_VAR, DEFAULT_LEARN_RATE, 'learning rate')
# mode
flags.DEFINE_enum(MODE_VAR, DEFAULT_MODE, ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
# output
flags.DEFINE_string(OUTPUT_VAR, OUTPUT_PATH, 'output for the model and images')
# preference file
flags.DEFINE_string( PREF_VAR, NO_PREF_PATH, 'prefences file path')
# tiny
flags.DEFINE_boolean( TINY_WEIGHTS_VAR, DEFAULT_WEIGHT_TYPE, 'yolov3 or yolov3-tiny')
# transfer type
flags.DEFINE_enum( TRANSFER_VAR, DFEAULT_TRANSFER_TYPE,
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
# validate image input
flags.DEFINE_string(VALID_IN_VAR, INPUT_IMAGE_PATH, 'path to image output')
# weights num
flags.DEFINE_integer(WEIGHTS_PATH_VAR, DEFAULT_WEIGHT_NUM, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')

########################
###     Scripts      ###
########################
# Create Classifier
flags.DEFINE_boolean( CONVERT_WEIGHT, False, 'weights')
# Create Classifier
flags.DEFINE_boolean( CREATE_CLASS_VAR, False, 'classifier')
# Detect Images
flags.DEFINE_boolean( DETECT_IMAGES_VAR, False, 'image_size')
# Export Core ML model
flags.DEFINE_boolean( EXPORT_COREML_VAR, False, 'coreML')
# Export TF Model
flags.DEFINE_boolean( EXPORT_Tf_MODEL_VAR, False, 'tf model')
# Generate TF Records
flags.DEFINE_boolean( GENRATE_TF_VAR, False, 'ygenerate tf')
# Train
flags.DEFINE_boolean( TRAIN_VAR, False, 'train')
# Sort Images
flags.DEFINE_boolean( SORT_IMAGES_VAR, False, 'sort images')

FLAGS(sys.argv)


########################## CHECK_VARIABLE_EXIST #############################
def check_preferences(var, flag, type):
    if FLAGS.pref != NO_PREF_PATH:
        var = var + ':'
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if var in line:
                    txt_input = line.split(":")[1]
                    # integer variable
                    if type == INT:
                        txt_input = re.sub("\D", "", txt_input)
                        if txt_input.isnumeric():
                            return int(txt_input)
                    # float variable
                    elif type == FLOAT:
                        txt_input = re.sub("\D", "", txt_input)
                        if txt_input.isfloat():
                            return float(txt_input)
                    # string/file variable
                    elif type == FILE:
                        txt_input = txt_input.replace(" ", "")
                        txt_input = txt_input.replace("\n", "")
                        if path.exists(txt_input):
                            return txt_input
                    # boolean variable
                    elif type == BOOL:
                        txt_input = is_tiny.lower()
                        if "true"in txt_input:
                            return True
                        elif "false" in txt_input:
                            return False
                    # list variable
                    else:
                        txt_input = txt_input.replace(" ", "")
                        txt_input = txt_input.replace("\n", "")
                        if txt_input in type:
                            return txt_input
    return flag

####### WEIGHTS PATH ######
def get_weights_path(is_tiny):
    if is_tiny:
        return TINY_PATH
    return YOLO_PATH


def get_help():
    if FLAGS.help != DEFAULT_HELP:
        print("\n\t\t\t////////////////////////////////")
        print("\t\t\t//     Workbench Help Menu    //")
        print("\t\t\t////////////////////////////////\n")
        print("Values:")
        print("\t--" + BATCH_SIZE_VAR + ": batch size of training...........................................(Default: " + str(DEFAULT_BATCH_SIZE) + ")")
        print("\t--" + CHECKPOINT_VAR + ": path that the checkpoint is saved...........................(Default: " + DEFAULT_CHECKPOINT_PATH + ")")
        print("\t--" + CLASSIFIERS_VAR + ": name of the classifier names file...............................(Default: " + CLASSIFIER_FILE + ")")
        print("\t--" + DATASET_TEST_VAR + ": path to test tf record file....................................(Default: " + TEST_TF_RECORD_PATH + ")")
        print("\t--" + DATASET_TRAIN_VAR + ": path to train tf record file..................................(Default: " + TRAIN_TF_RECORD_PATH + ")")
        print("\t--" + EPOCH_NUM_VAR + ": number of epochs used for training...................................(Default: " + str(DEFAULT_EPOCH_NUM) + ")")
        print("\t--" + IMAGE_SIZE_VAR + ": of your the images being trained on..............................(Default: " + str(DEFAULT_IMAGE_SIZE) + ")")
        print("\t--" + LEARN_RATE_VAR + ": learning rate of the training....................................(Default: " + str(DEFAULT_LEARN_RATE) + ")")
        print("\t--" + MODE_VAR + ": mode of the training...................................................(Default: " + DEFAULT_MODE + ")")
        print("\t\tOptions:")
        print("\t\t\t(1) fit: model.fit")
        print("\t\t\t(2) eager_fit: model.fit(run_eagerly=True")
        print("\t\t\t(3) eager_tf: custom GradientTape\n")
        print("\t--" + OUTPUT_VAR + ": location where tf and core ml model will be saved..............(Default: " + OUTPUT_PATH + ")")
        print("\t--" + PREF_VAR + ": file that contains preferences, this cannot be ran with other flags....(Default: " + NO_PREF_PATH + ")")
        print("\t--" + TINY_WEIGHTS_VAR + ": training with the tiny weights or not..........................(Default: " + str(DEFAULT_WEIGHT_TYPE) + ")")
        print("\t\tOptions:")
        print("\t\t\t(1) True: tiny_weights")
        print("\t\t\t(2) False: yolo3_weight\ns")
        print("\t--" + TRANSFER_VAR + " type of transfer used for training.............................(Default: " + DFEAULT_TRANSFER_TYPE + ")")
        print("\t\tOptions:")
        print("\t\t\t(1) none: Training from scratch")
        print("\t\t\t(2) darknet:  Transfer darknet")
        print("\t\t\t(3) no_output: Transfer all but output")
        print("\t\t\t(4) frozen: Transfer and freeze all")
        print("\t\t\t(5) fine_tune: Transfer all and freeze darknet only\n")
        print("\t--" + WEIGHTS_PATH_VAR + ": number of classes the weights file is trained on...............(Default: " +str( DEFAULT_WEIGHT_NUM) + ")")
        print("\t--" + VALID_IN_VAR + ": path to image(s) you want to test the new model on.....(Default: " + INPUT_IMAGE_PATH + ")")
        print("\n\nRunning specific scripts (can only run one):")
        print("\t--" + CONVERT_WEIGHT + ": converts the weight file and stops the program")
        print("\t--" + CREATE_CLASS_VAR + ": just generates the classifier file and stops the program")
        print("\t--" + DETECT_IMAGES_VAR  + ": just tests model on images and stops the program")
        print("\t--" + EXPORT_COREML_VAR + ": just exports the CoreML model and stops the program")
        print("\t--" + EXPORT_Tf_MODEL_VAR  + ": just exports the TensorFlow model and stops the program")
        print("\t--" + GENRATE_TF_VAR + ": just generates the tf records and stops the program")
        print("\t--" + TRAIN_VAR + ": just trains the model and stops the program")
        print("\t--" + SORT_IMAGES_VAR + ": sorts the image directory and stops the program")
        exit()
