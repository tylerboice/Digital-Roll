import os
import re
import sys

from absl import app, flags, logging
from absl.flags import FLAGS
from os import path

CWD_PATH = os.getcwd() + "/"

# IMAGE SET PATHS
IMAGES_PATH = CWD_PATH + "images/"
INPUT_IMAGE_PATH = IMAGES_PATH + "validate/"
TRAIN_IMAGE_PATH = IMAGES_PATH + "train/"
TEST_IMAGE_PATH = IMAGES_PATH + "test/"
VALIDATE_IMAGE_PATH = IMAGES_PATH + "validate/"

# DATA PATHS
DATA_PATH = CWD_PATH + "data" + "/"

CLASSIFIER_FILE = DATA_PATH + "classifier.names"
TRAIN_TF_RECORD_PATH = DATA_PATH + "train.tfrecord"
TEST_TF_RECORD_PATH = DATA_PATH + "test.tfrecord"

# CHECKPOINTS
CHECKPOINT_PATH = CWD_PATH + "checkpoints/"

# OUTPUT
OUTPUT_MODEL_PATH = CWD_PATH + "output/"

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
DEFAULT_IMAGE_SIZE = 416
DEFAULT_LEARN_RATE = .001
DEFAULT_MODE = 'fit'
DEFAULT_NUM_VAL_IMAGES = 3
DEFAULT_PREF_PATH = CWD_PATH + "preferences.txt"
DFEAULT_TRANSFER_TYPE = 'none'
DEFAULT_WEIGHT_NUM = 80
DEFAULT_WEIGHT_TYPE = False #False = yolov3 True = tiny

# PREFERENCE VARIABLE NAMES
BATCH_SIZE_VAR = "batch_size:"
CHECKPOINT_VAR = "checkpoint_path:"
CLASSIFIERS_VAR = "classifiers:"
DATASET_TEST_VAR = "dataset-test:"
DATASET_TRAIN_VAR = "dataset-train:"
EPOCH_NUM_VAR = "epochs:"
IMAGE_SIZE_VAR = "image_size:"
LEARN_RATE_VAR = "learn_rate:"
OUTPUT_MODEL_VAR = "output_model:"
MODE_VAR = "mode:"
TEST_IMAGE_VAR = 'validate_images:'
TINY_WEIGHTS_VAR = "tiny_weights:"
TRANSFER_VAR = "transfer_type:"
VALID_IN_VAR = "validate_image_input:"
WEIGHTS_CLASS_VAR = "Weighted_class:"
WEIGHTS_PATH_VAR = "weights_path:"



########################## FLAGS #################################
# batch_size
flags.DEFINE_integer('batch_size', DEFAULT_BATCH_SIZE, 'batch size')
# checkpoint path
flags.DEFINE_string('chkpnt_output', DEFAULT_CHECKPOINT_PATH, 'path to output')
# classifiers
flags.DEFINE_string('classifiers', CLASSIFIER_FILE, 'file of classifier names')
# dataset_test
flags.DEFINE_string('dataset_test', TEST_TF_RECORD_PATH, 'path to test tf record')
# dataset_train
flags.DEFINE_string('dataset_train', TRAIN_TF_RECORD_PATH, 'path to training tf record')
# ephoch num
flags.DEFINE_integer('epochs', DEFAULT_EPOCH_NUM, 'number of epochs')
# image_size
flags.DEFINE_integer('image_size', DEFAULT_IMAGE_SIZE, 'image size')
# learn rate
flags.DEFINE_float('learn_rate', DEFAULT_LEARN_RATE, 'learning rate')
# mode
flags.DEFINE_enum('mode', DEFAULT_MODE, ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
# output_model
flags.DEFINE_string('output_model', OUTPUT_MODEL_PATH, 'output for the model and images')
# preference file
flags.DEFINE_string('pref', NO_PREF_PATH, 'prefences file path')
# tiny
flags.DEFINE_boolean('tiny_weights', DEFAULT_WEIGHT_TYPE, 'yolov3 or yolov3-tiny')
# transfer type
flags.DEFINE_enum('transfer', DFEAULT_TRANSFER_TYPE,
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
# validate image input
flags.DEFINE_string('validate_input', INPUT_IMAGE_PATH, 'path to image output')
# weights num
flags.DEFINE_integer('weight_num_classes', DEFAULT_WEIGHT_NUM, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')

FLAGS(sys.argv)


########################## CHECK_VARIABLE_EXIST #############################
def check_preferences(var, flag, type):
    if FLAGS.pref != NO_PREF_PATH:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if var in line:
                    input = line.split(":")[1]
                    # integer variable
                    if type == INT:
                        input = re.sub("\D", "", input)
                        if input.isnumeric():
                            return int(input)
                    # float variable
                    elif type == FLOAT:
                        input = re.sub("\D", "", input)
                        if input.isfloat():
                            return float(input)
                    # string/file variable
                    elif type == FILE:
                        input = input.replace(" ", "")
                        input = input.replace("\n", "")
                        if path.exists(input):
                            return input
                    # boolean variable
                    elif type == BOOL:
                        input = is_tiny.lower()
                        if "true"in input:
                            return True
                        elif "false" in input:
                            return False
                    # list variable
                    else:
                        input = input.replace(" ", "")
                        input = input.replace("\n", "")
                        if input in type:
                            return input
    return flag

####### WEIGHTS PATH ######
def get_weights_path(is_tiny):
    if is_tiny:
        return TINY_PATH
    return YOLO_PATH
