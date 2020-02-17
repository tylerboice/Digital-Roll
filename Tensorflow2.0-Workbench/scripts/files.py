import os
import random
import shutil

from os import path
from scripts import defaults

CLASSIFIER_FILE = defaults.CLASSIFIER_FILE
IMAGES_PATH = defaults.IMAGES_PATH
MIN_IMAGES = defaults.MIN_IMAGES
OUTPUT_PATH = defaults.OUTPUT_PATH
TEST_IMAGE_PATH = defaults.TEST_IMAGE_PATH
TRAIN_IMAGE_PATH = defaults.TRAIN_IMAGE_PATH
VALIDATE_IMAGE_PATH = defaults.VALIDATE_IMAGE_PATH


def print_error(name, PATH):
    print(name +" path does not exist, please make sure you have a path at " + PATH )

########################## Checking FOR FILES #############################
# checks if all necessary files exist
def checkIfNecessaryPathsAndFilesExist():

    ####### IMAGE PATH #######
    if not os.path.exists(IMAGES_PATH):
        print_error("Image", IMAGE_PATH)

    ####### OUTPUT MODEL PATH #######
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    ####### TEST IMAGE PATH #######
    if not os.path.exists(TEST_IMAGE_PATH):
        os.mkdir(TEST_IMAGE_PATH)

    ####### TRAIN IMAGE PATH #######
    if not os.path.exists(TRAIN_IMAGE_PATH):
        os.mkdir(TRAIN_IMAGE_PATH)

    ####### VALIDATE IMAGE PATH #######
    if not os.path.exists(VALIDATE_IMAGE_PATH):
        os.mkdir(VALIDATE_IMAGE_PATH)


########################## SORT IMAGES #############################
# Takes all the images in the image folder and places them in test, train and validate
# Train = 90% of the images
# Test = 10% of the images
# Validate = takes user spicifed amount of files out of train
def sort_images(num_validate):

    # Method Variables
    total_images = 0
    file_count = 0
    train_images = 0
    unlabelled_files = []
    valid_images = []
    get_valid = False

    # For every file image in the image dir, check if it has an xml file and move it
    for filename in os.listdir(IMAGES_PATH):
        if '.png' in filename or '.jpg' in filename or '.jpeg' in filename:
            found_label = False
            xml_version = filename.split(".")[0] + ".xml"
            total_images += 1

            # move to test
            if total_images % 10 == 0:
                if path.exists(IMAGES_PATH + xml_version):
                    if not path.exists(TEST_IMAGE_PATH + filename) and not path.exists(TEST_IMAGE_PATH + xml_version):
                        shutil.move(IMAGES_PATH + filename, TEST_IMAGE_PATH)
                        shutil.move(IMAGES_PATH + xml_version, TEST_IMAGE_PATH)
                        found_label = True

            # move to train
            else:
                if path.exists(IMAGES_PATH + xml_version):
                    if not path.exists(TRAIN_IMAGE_PATH + filename) and not path.exists(TRAIN_IMAGE_PATH + xml_version):
                        shutil.move(IMAGES_PATH + filename, TRAIN_IMAGE_PATH)
                        shutil.move(IMAGES_PATH + xml_version, TRAIN_IMAGE_PATH)
                        found_label = True

            # if image was found but label was not:
            if found_label == False:
                unlabelled_files.append(filename)

    # count all image and .xml files in test
    for filename in os.listdir(TEST_IMAGE_PATH):
        if '.png' in filename or '.jpg' in filename or '.jpeg' in filename:
            total_images += 1

    # count all image and .xml files in train
    for filename in os.listdir(TRAIN_IMAGE_PATH):
        if '.png' in filename or '.jpg' in filename or '.jpeg' in filename:
            total_images += 1
            train_images += 1

    # print total image count
    print("\tTotal images = " + str(total_images))
    print("\tImages not labelled = " + str(len(unlabelled_files)))

    # total images must be greater than pre-defined count to train on
    if total_images < MIN_IMAGES:
        print("\n\nTensorflow needs at least " + str(MIN_IMAGES) + " images to train")
        exit()

    # if files are unlabelled, print them
    if len(unlabelled_files) != 0:
        print("The following images do not have a label:\n\t")
        print("\t" + unlabelled_files)

    # move all files in validate to train
    for file in os.listdir(VALIDATE_IMAGE_PATH):
        shutil.move(VALIDATE_IMAGE_PATH + file, TRAIN_IMAGE_PATH)

    # gather all valid images from train
    while len(valid_images) < num_validate:
        next_valid = random.randint(1, train_images)
        if next_valid not in valid_images:
            valid_images.append(next_valid)

    # move random valid images from train to validate
    file_count = 0
    for file in os.listdir(TRAIN_IMAGE_PATH):
        if '.png' in file or '.jpg' in file or '.jpeg' in file:
            file_count += 1
            xml_version = file.split(".")[0] + ".xml"
            if file_count in valid_images:
                shutil.move(TRAIN_IMAGE_PATH + file, VALIDATE_IMAGE_PATH)
                if path.exists(TRAIN_IMAGE_PATH + xml_version):
                    shutil.move(TRAIN_IMAGE_PATH + xml_version, VALIDATE_IMAGE_PATH)

########################## GET CLASSIFIERS #############################
# Reads all the xml files and gathers all the unique classifiers
def get_classifiers(data_dir):
    class_counter = 0
    classifiers = []
    for file in os.listdir(data_dir):
        if path.isfile(data_dir + file) == False:
            dir = data_dir + file + "/"
            nested_folder = get_classifiers(dir)
            if classifiers == None or classifiers == []:
                classifiers = nested_folder
            else:
                classifiers = classifiers + list(set(nested_folder) - set(classifiers))

        if ".xml" in file:
            with open(data_dir + file, "r") as f:
                for line in f.readlines():
                    if '<name>' in line:
                        name = line.replace("</name>", "")
                        name = name.replace("<name>", "")
                        name = name.replace("\t", "")
                        name = name.replace("\n", "")
                        if name not in classifiers:
                            classifiers.append(name)
                            classifiers.sort()
    return classifiers



########################## CREATE_CLASSIFIER_NAMES #############################
# takes in a list of all classifiers and writes to the CLASSIFIER_FILE each classifier
def create_classifier_file(classifiers):
    stored_lines = []
    class_counter = 0
    with open(CLASSIFIER_FILE, "w") as f:
        for classification in classifiers:
            class_counter += 1
            f.write(classification + "\n")


########################## GET_LAST_CHECKPOINT #############################
# gets the name of the last classifier from training
def get_last_checkpoint():
    last_checkpoint_num = 0
    last_checkpoint = ""
    for filename in os.listdir(defaults.CHECKPOINT_PATH):
        if 'tf.index' and 'train' in filename:
           if 'of' not in filename:
               current = filename.split(".")[0]
               current = current.split('_')[2]
               if last_checkpoint_num < int(current):
                   last_checkpoint_num = int(current)
                   last_checkpoint = defaults.CHECKPOINT_PATH + filename.split(".")[0] + ".tf"
    if last_checkpoint_num == 0:
        print("No chekpoint found")
        exit()
    return last_checkpoint


########################## GET NUM CLASSES ##########################
def get_num_classes(file):
    num_classes = 0
    if not os.path.exists(file):
        open(file, "w")
    with open(file, "r") as f:
        for line in f.readlines():
            if len(line.strip()) != 0 :
                num_classes = num_classes + 1
    return num_classes
