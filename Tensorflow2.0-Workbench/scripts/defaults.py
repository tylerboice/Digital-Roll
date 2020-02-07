import os
import re
import sys

from absl import app, flags, logging
from absl.flags import FLAGS
from os import path

CWD_PATH = os.getcwd() + "/"

# IMAGE SET PATHS
IMAGES_PATH = CWD_PATH + "images/"
TRAIN_IMAGE_PATH = IMAGES_PATH + "train/"
TEST_IMAGE_PATH = IMAGES_PATH + "test/"
VALIDATE_IMAGE_PATH = IMAGES_PATH + "validate/"

# DATA PATHS
DATA_PATH = CWD_PATH + "data" + "/"

CLASSIFIER_FILE = DATA_PATH + "classifier.names"
TRAIN_TF_RECORD_PATH = DATA_PATH + "train.tfrecord"
TEST_TF_RECORD_PATH = DATA_PATH + "test.tfrecord"

# HARD CODED VALUES
MIN_IMAGES = 50

TRANSFER_TYPES = ['none', 'darknet', 'no_output', 'frozen', 'fine_tune']

# Default values
DEFAULT_BATCH_SIZE = 4
DEFAULT_CHECKPOINT_PATH = CWD_PATH + 'checkpoints/yolov3.tf'
DEFAULT_EPOCH_NUM = 10
DEFAULT_IMAGE_SIZE = 416
DEFAULT_LEARN_RATE = 1e-3
DEFAULT_NUM_VAL_IMAGES = 3
DEFAULT_PREF_PATH = CWD_PATH + "preferences.txt"
DFEAULT_TRANSFER_TYPE = 'none'
DEFAULT_WEIGHT_NUM = 80
DEFAULT_WEIGHT_PATH = DATA_PATH + "yolov3.weights"
DEFAULT_WEIGHT_TYPE = False #False = yolov3 True = tiny

# PREFERENCE VARIABLE NAMES
BATCH_SIZE_VAR = "batch_size:"
CHECKPOINT_VAR = "checkpoint_path:"
EPOCH_NUM_VAR = "num_epochs:"
IMAGE_SIZE_VAR = "image_size:"
LEARN_RATE_VAR = "learn_rate:"
TEST_IMAGE_VAR = 'validate_images:'
TRANSFER_VAR = "transfer_type:"
WEIGHTS_PATH_VAR = "weights_path:"
WEIGHTS_TYPE_VAR = "weights_type:"

##################### GET PREFENCE FILE ##########################
flags.DEFINE_string('pref', DEFAULT_PREF_PATH, 'prefences file path')

def get_pref_path():
    if not os.path.exists(FLAGS.pref):
        print("No preference file found, using default values")
    return DEFAULT_PREF_PATH


PREFERENCES_PATH = DEFAULT_PREF_PATH
if not os.path.exists(PREFERENCES_PATH):
    print(PREFERENCES_PATH + " not found")


########################## SET PREFERENCES #############################
# checking for user preferences from file
####### GET BATCH SIZE #######
def get_batch_size():
    if os.path.exists(PREFERENCES_PATH):
        with open(PREFERENCES_PATH, "r") as f:
            for line in f.readlines():
                if BATCH_SIZE_VAR in line:
                    batch_size = line.split(":")[1]
                    batch_size = re.sub("\D", "", batch_size)
                    if batch_size.isnumeric():
                        return int(batch_size)
        print_var_not_found(BATCH_SIZE_VAR, DEFAULT_BATCH_SIZE)
    return DEFAULT_BATCH_SIZE

####### GET CHECKPOINT #######
def get_checkponit_path():
    if os.path.exists(PREFERENCES_PATH):
        with open(PREFERENCES_PATH, "r") as f:
            for line in f.readlines():
                if CHECKPOINT_VAR in line:
                    checkpoint = line.split(":")[1]
                    checkpoint = checkpoint.replace(" ", "")
                    checkpoint = checkpoint.replace("\n", "")
                    if path.exists(checkpoint):
                        return checkpoint
    return DEFAULT_CHECKPOINT_PATH

####### GET EPCH NUM #######
def get_epoch_num():
    if os.path.exists(PREFERENCES_PATH):
        with open(PREFERENCES_PATH, "r") as f:
            for line in f.readlines():
                if EPOCH_NUM_VAR in line:
                    epoch_num = line.split(":")[1]
                    epoch_num = re.sub("\D", "", epoch_num)
                    if epoch_num.isnumeric():
                        return int(epoch_num)
        print_var_not_found(EPOCH_NUM_VAR, DEFAULT_EPOCH_NUM)
    return DEFAULT_EPOCH_NUM

####### GET IMAGE_SIZE #######
def get_image_size():
    if os.path.exists(PREFERENCES_PATH):
        with open(PREFERENCES_PATH, "r") as f:
            for line in f.readlines():
                if IMAGE_SIZE_VAR in line:
                    image_size = line.split(":")[1]
                    image_size = re.sub("\D", "", image_size)
                    if image_size.isnumeric():
                        return int(image_size)
        print_var_not_found(IMAGE_SIZE_VAR, DEFAULT_IMAGE_SIZE)
    return DEFAULT_IMAGE_SIZE

####### GET LEARN RATE #######
def get_learn_rate():
    if os.path.exists(PREFERENCES_PATH):
        with open(PREFERENCES_PATH, "r") as f:
            for line in f.readlines():
                if LEARN_RATE_VAR in line:
                    learn_rate = line.split(":")[1]
                    learn_rate = learn_rate.replace(" ", "")
                    learn_rate = learn_rate.replace("\n", "")
                    if float(learn_rate) > .1 or float(learn_rate) <= 0:
                        print("Learn rate should not be > .1 or <= 0" + "Restoreing to default " + DEFAULT_LEARN_RATE)
                        return DEFAULT_LEARN_RATE
                    return float(learn_rate)
        print_var_not_found(LEARN_RATE_VAR, DEFAULT_LEARN_RATE)
    return DEFAULT_LEARN_RATE

####### GET NUM CLASSES #######
def get_num_classes():
    num_classes = 0
    with open(CLASSIFIER_FILE, "r") as f:
        for line in f.readlines():
            if len(line.strip()) != 0 :
                num_classes = num_classes + 1
    return num_classes


####### GET TRANSFER TYPE ######
def get_transfer_type():
    if os.path.exists(PREFERENCES_PATH):
        with open(PREFERENCES_PATH, "r") as f:
            for line in f.readlines():
                if TRANSFER_VAR in line:
                    transfer = line.split(":")[1]
                    transfer = transfer.replace(" ", "")
                    transfer =transfer.replace("\n", "")
                    if transfer in TRANSFER_TYPES:
                        return transfer
        print_var_not_found(TRANSFER_VAR, DFEAULT_TRANSFER_TYPE)
    return DFEAULT_TRANSFER_TYPE

####### GET VALID IMAGE COUNT #######
def get_valid_image_num():
    if os.path.exists(PREFERENCES_PATH):
        with open(PREFERENCES_PATH, "r") as f:
            for line in f.readlines():
                if TEST_IMAGE_VAR in line:
                    get_valid_image_num = line.split(":")[1]
                    get_valid_image_num = re.sub("\D", "", get_valid_image_num)
                    if get_valid_image_num.isnumeric():
                        return int(get_valid_image_num)
        print_var_not_found(TEST_IMAGE_VAR, DEFAULT_NUM_VAL_IMAGES)
    return DEFAULT_NUM_VAL_IMAGES

####### GET WEIGHTS PATH #######
def get_weights_path():
    opened = False
    if os.path.exists(PREFERENCES_PATH):
        with open(PREFERENCES_PATH, "r") as f:
            for line in f.readlines():
                opened = True
                if WEIGHTS_PATH_VAR in line:
                    weights = line.split(":")[1]
                    weights = weights.replace("\n", "")
                    weights_path = CWD_PATH + weights.replace(" ", "")
                    if path.exists(weights_path):
                        return weights_path
                    else:
                        print( weights_path + " does not exist")
        if not opened:
            print_var_not_found(WEIGHTS_PATH_VAR, DEFAULT_WEIGHT_PATH)
    if path.exists(DEFAULT_WEIGHT_PATH):
        return DEFAULT_WEIGHT_PATH
    else:
        print("No valid weights, check to make sure you have a weights file")
        exit()

####### GET WEIGHTS TYPE #######
def is_tiny_weight():
    if os.path.exists(PREFERENCES_PATH):
        with open(PREFERENCES_PATH, "r") as f:
            for line in f.readlines():
                if WEIGHTS_TYPE_VAR in line:
                    weight_type = line.split(":")[1]
                    weight_type = weight_type.replace(" ", "") + CWD_PATH
                    if "yolo3"in weight_type :
                        return False
                    elif "yolo-tiny"in weight_type :
                        return True
        print_var_not_found(WEIGHTS_TYPE_VAR, DEFAULT_WEIGHT_TYPE)
        return DEFAULT_WEIGHT_TYPE

def print_var_not_found(var_name, default):
    print("\t" + var_name + " variable not found in preferences file,  reverting to default " + default)

# batch_size
flags.DEFINE_integer('batch_size', get_batch_size(), 'batch size')
# checkpoint path
flags.DEFINE_string('output', get_checkponit_path(), 'path to output')
# ephoch num
flags.DEFINE_integer('epochs', get_epoch_num(), 'number of epochs')
# image_size
flags.DEFINE_integer('size', get_image_size(), 'image size')
# learn rate
flags.DEFINE_float('learning_rate', get_learn_rate(), 'learning rate')
# mode
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
# num_classes
flags.DEFINE_integer('num_classes', get_num_classes(), 'number of classes in the model')
# tiny_weights
flags.DEFINE_boolean('tiny', is_tiny_weight(), 'yolov3 or yolov3-tiny')
# transfer type
flags.DEFINE_enum('transfer', get_transfer_type(),
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
# weights num
flags.DEFINE_integer('weights_num_classes', DEFAULT_WEIGHT_NUM, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
# weights_path
flags.DEFINE_string('weights', get_weights_path(), 'path to weights file')
