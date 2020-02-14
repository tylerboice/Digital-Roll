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

TRANSFER_TYPES = ['none', 'darknet', 'no_output', 'frozen', 'fine_tune']
YOLO_PATH = DATA_PATH + "yolov3.weights"
TINY_PATH = DATA_PATH + "yolov3-tiny.weights"

# Default values
DEFAULT_BATCH_SIZE = 4
DEFAULT_CHECKPOINT_PATH = CWD_PATH + 'checkpoints/yolov3.tf'
DEFAULT_EPOCH_NUM = 10
DEFAULT_IMAGE_SIZE = 416
DEFAULT_LEARN_RATE = .001
DEFAULT_NUM_VAL_IMAGES = 3
DEFAULT_PREF_PATH = CWD_PATH + "preferences.txt"
DFEAULT_TRANSFER_TYPE = 'none'
DEFAULT_WEIGHT_NUM = 80
DEFAULT_WEIGHT_TYPE = False #False = yolov3 True = tiny

# PREFERENCE VARIABLE NAMES
BATCH_SIZE_VAR = "batch_size:"
CHECKPOINT_VAR = "checkpoint_path:"
DATASET_TEST_VAR = "dataset-test:"
DATASET_TRAIN_VAR = "dataset-train:"
EPOCH_NUM_VAR = "num_epochs:"
IMAGE_SIZE_VAR = "image_size:"
OUTPUT_MODEL_VAR = "output_model:"
LEARN_RATE_VAR = "learn_rate:"
TEST_IMAGE_VAR = 'validate_images:'
TRANSFER_VAR = "transfer_type:"
VALID_IN_VAR = "validate_image_input:"
WEIGHTS_CLASS_VAR = "Weighted_class:"
WEIGHTS_PATH_VAR = "weights_path:"
WEIGHTS_TYPE_VAR = "tiny_weights:"


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
flags.DEFINE_integer('size', DEFAULT_IMAGE_SIZE, 'image size')
# learn rate
flags.DEFINE_float('learning_rate', DEFAULT_LEARN_RATE, 'learning rate')
# mode
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
# output_model
flags.DEFINE_string('output_model', OUTPUT_MODEL_PATH, 'output for the model and images')
# preference file
flags.DEFINE_string('pref', DEFAULT_PREF_PATH, 'prefences file path')
# tiny
flags.DEFINE_boolean('tiny', DEFAULT_WEIGHT_TYPE, 'yolov3 or yolov3-tiny')
# transfer type
flags.DEFINE_enum('transfer', DFEAULT_TRANSFER_TYPE,
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
# validate image input
flags.DEFINE_string('image_output', INPUT_IMAGE_PATH, 'path to image output')
# weights num
flags.DEFINE_integer('weights_num_classes', DEFAULT_WEIGHT_NUM, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')

FLAGS(sys.argv)


########################## SET PREFERENCES #############################
# checking for user preferences from file
####### GET BATCH SIZE #######
def get_batch_size():
    if DEFAULT_PREF_PATH != FLAGS.pref:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if BATCH_SIZE_VAR in line:
                    batch_size = line.split(":")[1]
                    batch_size = re.sub("\D", "", batch_size)
                    if batch_size.isnumeric():
                        return int(batch_size)
        print_var_not_found(BATCH_SIZE_VAR, FLAGS.batch_size)
    return FLAGS.batch_size

####### GET CHECKPOINT #######
def get_checkpoint_path():
    if DEFAULT_PREF_PATH != FLAGS.pref:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if CHECKPOINT_VAR in line:
                    checkpoint = line.split(":")[1]
                    checkpoint = checkpoint.replace(" ", "")
                    checkpoint = checkpoint.replace("\n", "")
                    if path.exists(checkpoint):
                        return checkpoint
    return FLAGS.chkpnt_output

####### GET DATASET TEST #######
def get_dataset_test():
    if DEFAULT_PREF_PATH != FLAGS.pref:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if DATASET_TEST_VAR in line:
                    dataset = line.split(":")[1]
                    dataset = dataset.replace(" ", "")
                    dataset = dataset.replace("\n", "")
                    if path.exists(dataset):
                        return dataset
    return FLAGS.dataset_test

####### GET DATASET TRAIN #######
def get_dataset_train():
    if DEFAULT_PREF_PATH != FLAGS.pref:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if DATASET_TRAIN_VAR in line:
                    dataset = line.split(":")[1]
                    dataset = dataset.replace(" ", "")
                    dataset = dataset.replace("\n", "")
                    if path.exists(dataset):
                        return dataset
    return FLAGS.dataset_train

####### GET EPCH NUM #######
def get_epoch_num():
    if DEFAULT_PREF_PATH != FLAGS.pref:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if EPOCH_NUM_VAR in line:
                    epoch_num = line.split(":")[1]
                    epoch_num = re.sub("\D", "", epoch_num)
                    if epoch_num.isnumeric():
                        return int(epoch_num)
        print_var_not_found(EPOCH_NUM_VAR, FLAGS.epochs)
    return FLAGS.epochs

####### GET IMAGE_SIZE #######
def get_image_size():
    if DEFAULT_PREF_PATH != FLAGS.pref:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if IMAGE_SIZE_VAR in line:
                    image_size = line.split(":")[1]
                    image_size = re.sub("\D", "", image_size)
                    if image_size.isnumeric():
                        return int(image_size)
        print_var_not_found(IMAGE_SIZE_VAR, FLAGS.image_size)
    return FLAGS.size

####### GET LEARN RATE #######
def get_learn_rate():
    if DEFAULT_PREF_PATH != FLAGS.pref:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if LEARN_RATE_VAR in line:
                    learn_rate = line.split(":")[1]
                    learn_rate = learn_rate.replace(" ", "")
                    learn_rate = learn_rate.replace("\n", "")
                    if float(learn_rate) > .1 or float(learn_rate) <= 0:
                        print("Learn rate should not be > .1 or <= 0" + "Restoreing to default " + DEFAULT_LEARN_RATE)
                        return DEFAULT_LEARN_RATE
                    return float(learn_rate)
        print_var_not_found(LEARN_RATE_VAR, FLAGS.learn_rate)
    return FLAGS.learning_rate

####### GET NUM CLASSES #######
def get_num_classes(file):
    num_classes = 0
    with open(file, "r") as f:
        for line in f.readlines():
            if len(line.strip()) != 0 :
                num_classes = num_classes + 1
    return num_classes

####### TINY or NOT ##########
def get_model_output():
    if DEFAULT_PREF_PATH != FLAGS.pref:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if OUTPUT_MODEL_VAR in line:
                    output = line.split(":")[1]
                    output = output.lower()
                    if os.exist(output):
                        return output
    return FLAGS.output_model

####### TINY or NOT ##########
def is_tiny_weight():
    if DEFAULT_PREF_PATH != FLAGS.pref:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if WEIGHTS_TYPE_VAR in line:
                    is_tiny = line.split(":")[1]
                    is_tiny = is_tiny.lower()
                    if "true"in is_tiny:
                        return True
                    else:
                        return False
    return FLAGS.tiny

####### GET TRANSFER TYPE ######
def get_transfer_type():
    if DEFAULT_PREF_PATH != FLAGS.pref:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if TRANSFER_VAR in line:
                    transfer = line.split(":")[1]
                    transfer = transfer.replace(" ", "")
                    transfer =transfer.replace("\n", "")
                    if transfer in TRANSFER_TYPES:
                        return transfer
        print_var_not_found(TRANSFER_VAR, FLAGS.transfer)
    return FLAGS.transfer

####### GET VALID IMAGE COUNT #######
def get_valid_image_num():
    if DEFAULT_PREF_PATH != FLAGS.pref:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if TEST_IMAGE_VAR in line:
                    get_valid_image_num = line.split(":")[1]
                    get_valid_image_num = re.sub("\D", "", get_valid_image_num)
                    if get_valid_image_num.isnumeric():
                        return int(get_valid_image_num)
        print_var_not_found(TEST_IMAGE_VAR, DEFAULT_NUM_VAL_IMAGES)
    return DEFAULT_NUM_VAL_IMAGES

####### GET VALID IMAGE OUTPUT #######
def get_valid_image_input():
    if DEFAULT_PREF_PATH != FLAGS.pref:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if VALID_IN_VAR in line:
                    output = line.split(":")[1]
                    output = output.replace(" ", "")
                    output = output.replace("\n", "")
                    if path.exists(output):
                        return output
    return FLAGS.image_output

####### GET_VALID_TEST_IMAGE ######
def get_valid_test_image():
    for file in VALIDATE_IMAGE_PATH:
       if ".jpg" in file:
            return file
    print("No files to test on, place test files in " + VALIDATE_IMAGE_PATH)
    exit()


####### WEIGHTS PATH ######
def get_weights_path():
    if is_tiny_weight():
        return TINY_PATH
    else:
        return YOLO_PATH

####### GET WEIGHTS CLASS #######
def get_weight():
    if DEFAULT_PREF_PATH != FLAGS.pref:
        with open(FLAGS.pref, "r") as f:
            for line in f.readlines():
                if WEIGHTS_CLASS_VAR in line:
                    classes = line.split(":")[1]
                    classes = re.sub("\D", "", classes)
                    if classes.isnumeric():
                        return int(classes)
    return FLAGS.weights_num_classes

####### PINT STATEMENT ##########
def print_var_not_found(var_name, default):
    print("\t" + var_name + " variable not found in preferences file,  reverting to default " + default)

########### PRnit Preferences ##########3
def print_pref():

    print("\tBatch Size............. " + str(get_batch_size()))
    print("\tCheckpoint Output...... " + get_checkpoint_path())
    print("\tClassifier file........ " + FLAGS.classifiers)
    print("\tDataSet-test........... " + get_dataset_test())
    print("\tDataSet-train.......... " + get_dataset_train())
    print("\tEpochs................. " + str(get_epoch_num()))
    print("\tLearning Rate.......... " + str(get_learn_rate()))
    print("\tMode................... " + FLAGS.mode)
    print("\tNumber of Classes...... " + str(get_num_classes(FLAGS.classifiers)))
    print("\tOutput Model........... " + get_model_output())
    print("\tPreference File........ " + FLAGS.pref)
    print("\tSize of Image.......... " + str(get_image_size()))
    print("\tTiny Weights........... " + str(is_tiny_weight()))
    print("\tTransfer............... " + get_transfer_type())
    print("\tValidate Image Input... " + get_valid_image_input())
    print("\tWeighted Classes....... " + str(get_weight()))
    print("\tWeights Path........... " + get_weights_path())
