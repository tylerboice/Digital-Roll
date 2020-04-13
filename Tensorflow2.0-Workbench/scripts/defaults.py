import os
import re
import sys

from absl import app, flags, logging
from absl.flags import FLAGS
from os import path

CWD_PATH = os.getcwd().replace("\\", "/") + "/"

# DATA PATHS
DATA_PATH = os.path.join(CWD_PATH , "data/")
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

# IMAGE SET PATHS
IMAGES_PATH = os.path.join(CWD_PATH , "images/")
TRAIN_IMAGE_PATH = os.path.join(IMAGES_PATH , "train/")
TEST_IMAGE_PATH = os.path.join(IMAGES_PATH , "test/")
VALIDATE_IMAGE_PATH = os.path.join(IMAGES_PATH , "validate/")


# WEIGHTS PATHS
YOLO_PATH = os.path.join(DATA_PATH + "yolov3.weights")
TINY_PATH = os.path.join(DATA_PATH + "yolov3-tiny.weights")

MIN_IMAGES = 3
DEFAULT_LEARN_RATE = .001

# HARD CODED VALUES
BOOL = -999
INT = -998
FILE = -997
FLOAT = -996

#\\\\\\\\\\\\\\\\\\\\\
#\\     VALUES      \\
#\\\\\\\\\\\\\\\\\\\\\
################################ BATCH SIZE #####################################
BATCH_SIZE_VAR = "batch_size"
DEFAULT_BATCH_SIZE = 4
flags.DEFINE_integer( BATCH_SIZE_VAR, DEFAULT_BATCH_SIZE, 'batch size')


################################ CLASSIFIERS ####################################
CLASSIFIERS_VAR = "classifiers"
CLASSIFIER_FILE = os.path.join(DATA_PATH , "classifier.names")
flags.DEFINE_string( CLASSIFIERS_VAR, CLASSIFIER_FILE, 'file of classifier names')


############################### DATASET_TEST ####################################
DATASET_TEST_VAR = "dataset_test"
TEST_TF_RECORD_PATH = os.path.join(DATA_PATH , "test.tfrecord")
flags.DEFINE_string( DATASET_TEST_VAR, TEST_TF_RECORD_PATH, 'path to test tf record')


############################### DATASET_TRAIN ###################################
DATASET_TRAIN_VAR = "dataset_train"
TRAIN_TF_RECORD_PATH = os.path.join(DATA_PATH , "train.tfrecord")
flags.DEFINE_string( DATASET_TRAIN_VAR, TRAIN_TF_RECORD_PATH, 'path to training tf record')


################################# EPOCHS ########################################
EPOCH_NUM_VAR = "epochs"
DEFAULT_EPOCH_NUM = 10
flags.DEFINE_integer( EPOCH_NUM_VAR, DEFAULT_EPOCH_NUM, 'number of epochs')


################################ IMAGE SIZE #####################################
IMAGE_SIZE_VAR = "image_size"
DEFAULT_IMAGE_SIZE = 224
flags.DEFINE_integer( IMAGE_SIZE_VAR, DEFAULT_IMAGE_SIZE, 'image size')


############################## MAX CHECKPOINTS ##################################
MAX_CHECK_VAR = "max_checkpoints"
DEFAULT_MAX_CHECK = 10
flags.DEFINE_integer( MAX_CHECK_VAR, DEFAULT_MAX_CHECK, 'maximum checkpoints')


############################## MAX SESSIONS #####################################
MAX_SESS_VAR = "max_sessions"
DEFAULT_MAX_SESS = 5
flags.DEFINE_integer( MAX_SESS_VAR, DEFAULT_MAX_SESS, 'maximum checkpoints')


################################# MODE ##########################################
MODE_VAR = "mode"
DEFAULT_MODE = "fit"
MODE_OPTIONS = ['fit', 'eager_fit', 'eager_tf']
flags.DEFINE_enum(MODE_VAR, DEFAULT_MODE, MODE_OPTIONS,
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')


################################ OUTPUT #########################################
OUTPUT_VAR = "output"
OUTPUT_PATH = os.path.join(CWD_PATH , "current_session/")
flags.DEFINE_string(OUTPUT_VAR, OUTPUT_PATH, 'output for the model and images')


############################ PREFERENCES PATH ###################################
PREF_VAR = "pref"
NO_PREF_PATH = "none"
flags.DEFINE_string( PREF_VAR, NO_PREF_PATH, 'prefences file path')

############################ SESSIONS PATH ###################################
SAVED_SESS_VAR = "sessions"
SAVED_SESS_PATH = os.path.join(CWD_PATH , "saved_sessions/")
flags.DEFINE_string(SAVED_SESS_VAR, SAVED_SESS_PATH, 'location to save sessions')

############################ TINY WEIGHTS #######################################
TINY_WEIGHTS_VAR = "tiny_weights"
DEFAULT_WEIGHT_TYPE = False  # False = yolov3 True = tiny
flags.DEFINE_boolean( TINY_WEIGHTS_VAR, DEFAULT_WEIGHT_TYPE, 'yolov3 or yolov3-tiny')


############################## TRANSFER #########################################
TRANSFER_VAR = "transfer"
DFEAULT_TRANSFER_TYPE = 'none'
TRANSFER_OPTIONS = ['none', 'darknet', 'no_output', 'frozen', 'fine_tune']
flags.DEFINE_enum( TRANSFER_VAR, DFEAULT_TRANSFER_TYPE, TRANSFER_OPTIONS,
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')

############################## VALID IMAGE EXTRACTION ##################################
VALID_IMGS_VAR = "val_img_num"
DEFAULT_VALID_IMGS = 5
flags.DEFINE_integer( VALID_IMGS_VAR, DEFAULT_VALID_IMGS, 'images to extract')

########################### VALID IMAGE PATH ####################################
VALID_IN_VAR = "val_image_path"
INPUT_IMAGE_PATH = os.path.join(IMAGES_PATH , "validate/")
flags.DEFINE_string(VALID_IN_VAR, INPUT_IMAGE_PATH, 'path to image output')


########################## WEIGHTED NUM CLASS ###################################
WEIGHTS_NUM_VAR = "weighted_classes"
DEFAULT_WEIGHT_NUM = 80
flags.DEFINE_integer(WEIGHTS_NUM_VAR, DEFAULT_WEIGHT_NUM, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')

########################## WEIGHTED FILE PATH ###################################
WEIGHTS_PATH_VAR = "weights"
DEFAULT_WEIGHT_PATH = YOLO_PATH
# this flag is placed in preferences

########################
###     Scripts      ###
########################
# Create Weights
CONVERT_WEIGHT = "convert_weight"
flags.DEFINE_boolean( CONVERT_WEIGHT, False, 'weights')

# Create Classifier
CREATE_CLASS_VAR = "create_class_file"
flags.DEFINE_boolean( CREATE_CLASS_VAR, False, 'classifier')

# Detect Images
DETECT_IMAGES_VAR = "detect_img"
flags.DEFINE_boolean( DETECT_IMAGES_VAR, False, 'image_size')

# Export Core ML model
EXPORT_COREML_VAR = "core_ml"
flags.DEFINE_boolean( EXPORT_COREML_VAR, False, 'coreML')

# Export TF Model
EXPORT_Tf_MODEL_VAR = "tf_model"
flags.DEFINE_boolean( EXPORT_Tf_MODEL_VAR, False, 'tf model')

# Generate TF Records
GENRATE_TF_VAR = "generate_tf"
flags.DEFINE_boolean( GENRATE_TF_VAR, False, 'generate tf')

# Sort Images
SORT_IMAGES_VAR = "sort_images"
flags.DEFINE_boolean( SORT_IMAGES_VAR, False, 'sort images')

# Train
TRAIN_VAR = "train"
flags.DEFINE_boolean( TRAIN_VAR, False, 'train')


FLAGS(sys.argv)


########################## CHECK_PREFERENCES #############################
# Description: checks pref file for changes in variables
# Parameters: var - String - varable name
#             flag - name of the flags.DEFINE
#             type - int - number id for value type of the flag (Int, Bool, Sting, Doublg)
# Return: variable value
def check_preferences(var, flag, type):
    if FLAGS.pref != NO_PREF_PATH:
        var = var + '='
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
# Description: returns tiny weight path if true, returns yolov3.weights if false
# Parameters: is_tiny - Boolean - True if tiny, False if not
# Return: String - path to the weights file
def get_weights_path(is_tiny):
    if is_tiny:
        return TINY_PATH
    return YOLO_PATH
