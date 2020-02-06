import os
import re
from absl import app, flags, logging
from absl.flags import FLAGS

# current directory
os.chdir("../")

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
YOLO_TRAINED_NUM_CLASSES = 80
TRANSFER_TYPES = ['none', 'darknet', 'no_output', 'frozen', 'fine_tune']

# Default values
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCH_NUM = 10
DEFAULT_IMAGE_SIZE = 416
DEFAULT_LEARN_RATE = 1e-3
DEFAULT_NUM_VAL_IMAGES = 3
DEFAULT_PREF_PATH = CWD_PATH + "preferences.txt"
DFEAULT_TRANSFER_TYPE = 'fine_tune'
DEFAULT_WEIGHT_PATH = DATA_PATH + "yolov3.weights"
DEFAULT_WEIGHT_TYPE = False #False = yolov3 True = tiny

# PREFERENCE VARIABLE NAMES
BATCH_SIZE_VAR = "batch_size:"
EPOCH_NUM_VAR = "num_epochs:"
IMAGE_SIZE_VAR = "image_size:"
LEARN_RATE_VAR = "learn_rate:"
TEST_IMAGE_VAR = 'validate_images:'
TRANSFER_VAR = "transfer_type:"
WEIGHTS_PATH_VAR = "weights_path:"
WEIGHTS_TYPE_VAR = "weights_type:"

########################## SET PREFERENCES #############################
# checking for user preferences from file
####### GET BATCH SIZE #######
def get_batch_size(pref_file):
    if os.path.exists(pref_file):
        with open(pref_file, "r") as f:
            for line in f.readlines():
                if BATCH_SIZE_VAR in line:
                    batch_size = line.split(":")[1]
                    batch_size = re.sub("\D", "", batch_size)
                    if batch_size.isnumeric():
                        return int(batch_size)
    print("\tNo " + BATCH_SIZE_VAR + " variable in preferences, reverting to default: " + str(DEFAULT_BATCH_SIZE))
    return DEFAULT_BATCH_SIZE

####### GET EPCH NUM #######
def get_epoch_num(pref_file):
    if os.path.exists(pref_file):
        with open(pref_file, "r") as f:
            for line in f.readlines():
                if EPOCH_NUM_VAR in line:
                    epoch_num = line.split(":")[1]
                    epoch_num = re.sub("\D", "", epoch_num)
                    if epoch_num.isnumeric():
                        return int(epoch_num)
    print("\tNo " + EPOCH_NUM_VAR + " variable in preferences, reverting to default: " + str(DEFAULT_EPOCH_NUM))
    return DEFAULT_EPOCH_NUM

####### GET IMAGE_SIZE #######
def get_image_size(pref_file):
    if os.path.exists(pref_file):
        with open(pref_file, "r") as f:
            for line in f.readlines():
                if IMAGE_SIZE_VAR in line:
                    image_size = line.split(":")[1]
                    image_size = re.sub("\D", "", image_size)
                    if image_size.isnumeric():
                        return int(image_size)
    print("\tNo " + IMAGE_SIZE_VAR + " variable in preferences, reverting to default: " + str(DEFAULT_IMAGE_SIZE))
    return DEFAULT_IMAGE_SIZE

####### GET LEARN RATE #######
def get_learn_rate(pref_file):
    if os.path.exists(pref_file):
        with open(pref_file, "r") as f:
            for line in f.readlines():
                if LEARN_RATE_VAR in line:
                    learn_rate = line.split(":")[1]
                    learn_rate = learn_rate.replace(" ", "")
                    if learn_rate.isnumeric():
                        if int(learn_rate) > .1 or int(learn_rate) <= 0:
                            print("Learn rate should not be > .1 or <= 0")
                            exit()
                        return int(learn_rate)
    print("\tNo " + LEARN_RATE_VAR + " variable in preferences, reverting to default: " + str(DEFAULT_LEARN_RATE))
    return DEFAULT_LEARN_RATE

####### GET NUM CLASSES #######
def get_num_classes(names_file_path):
    num_classes = 0
    with open(names_file_path, "r") as f:
        for line in f.readlines():
            if len(line.strip()) != 0 :
                num_classes = num_classes + 1
    return num_classes

####### GET PREFERENCE FILE #######
def get_pref_path():
    if os.path.exists(DEFAULT_PREF_PATH):
        print("Preference path does not exist, using " + DEFAULT_PREF_PATH)
    else:
        print("No preference file found, using default values")
    return DEFAULT_PREF_PATH

####### GET TRANSFER TYPE ######
def get_transfer_type(pref_file):
    if os.path.exists(pref_file):
        with open(pref_file, "r") as f:
            for line in f.readlines():
                if TRANSFER_VAR in line:
                    transfer = line.split(":")[1]
                    transfer = transfer.replace(" ", "")
                    if transfer in TRANSFER_TYPES:
                        return transfer
    print("\tNo " + TRANSFER_VAR + " variable in preferences or incorrect input, reverting to default: " + DFEAULT_TRANSFER_TYPE)
    return DFEAULT_TRANSFER_TYPE

####### GET VALID IMAGE COUNT #######
def get_valid_image_num(pref_file):
    if os.path.exists(pref_file):
        with open(pref_file, "r") as f:
            for line in f.readlines():
                if TEST_IMAGE_VAR in line:
                    get_valid_image_num = line.split(":")[1]
                    get_valid_image_num = re.sub("\D", "", get_valid_image_num)
                    if get_valid_image_num.isnumeric():
                        return int(get_valid_image_num)
    print("\tNo " + TEST_IMAGE_VAR + " variable in preferences, reverting to default: " + str(DEFAULT_NUM_VAL_IMAGES))
    return DEFAULT_NUM_VAL_IMAGES

####### GET WEIGHTS PATH #######
def get_weight_path(pref_file):
    if os.path.exists(pref_file):
        with open(pref_file, "r") as f:
            for line in f.readlines():
                if WEIGHTS_PATH_VAR in line:
                    weights = line.split(":")[1]
                    weights_path = weights.replace(" ", "") + CWD_PATH
                    if os.path.exists(weights_path):
                        return weights_path
    print("\tNo " + WEIGHTS_PATH_VAR + " variable in preferences or path does not exist, reverting to default: " + DEFAULT_WEIGHT_PATH)
    return DEFAULT_WEIGHT_PATH

####### GET WEIGHTS TYPE #######
def is_tiny_weight(pref_file):
    if os.path.exists(pref_file):
        with open(pref_file, "r") as f:
            for line in f.readlines():
                if WEIGHTS_TYPE_VAR in line:
                    weight_type = line.split(":")[1]
                    weight_type = weight_type.replace(" ", "") + CWD_PATH
                    if "yolo3"in weight_type :
                        return False
                    elif "yolo-tiny"in weight_type :
                        return True
    print("\tNo " + WEIGHTS_TYPE_VAR + " variable in preferences or path does not exist, reverting to default")
