import time
import glob
import csv
import os
import re
import hashlib
import random
import pandas as pd
import shutil

from absl import app, flags, logging
from absl.flags import FLAGS
from os import path
import tensorflow as tf
import lxml.etree
import tqdm

CWD_PATH = os.path.join(os.getcwd() + '/')

# DATA SET PATHS
IMAGES_PATH = os.path.join(CWD_PATH, "images/")
TRAIN_IMAGE_PATH = os.path.join(IMAGES_PATH, "train/")
TEST_IMAGE_PATH = os.path.join(IMAGES_PATH, "test/")
VALIDATE_IMAGE_PATH = os.path.join(IMAGES_PATH, "validate/")

# DATA PATHS
DATA_PATH = os.path.join(CWD_PATH + "data/")

CLASSIFIER_FILE = os.path.join(DATA_PATH, "classifier.names")
PREFERENCES_PATH = os.path.join(CWD_PATH, "preferences.txt")
TRAIN_TF_RECORD_PATH = os.path.join(DATA_PATH, "train.tfrecord")
TEST_TF_RECORD_PATH = os.path.join(DATA_PATH, "test.tfrecord")

##### STATIC VARIABLES ####
DEFAULT_NUM_VAL_IMAGES = 3
MIN_IMAGES = 50
TEST_IMAGE_VAR = 'validate_images:'

flags.DEFINE_enum('split', 'train', [
                  'train', 'val'], 'specify train or val split')
flags.DEFINE_string('output_file', './data/new.tfrecord', 'outpot IMAGES')
flags.DEFINE_string('classes', './data/voc2012.names', 'classes file')

########################## CHECK FOR FILES #############################
# checks if all necessary files exist
def checkIfNecessaryPathsAndFilesExist():

    ####### IMAGE PATH #######
    if not os.path.exists(IMAGES_PATH):
        print(IMAGES_PATH)

    ####### TEST IMAGE PATH #######
    if not os.path.exists(TEST_IMAGE_PATH):
        os.mkdir(TEST_IMAGE_PATH)

    ####### TRAIN IMAGE PATH #######
    if not os.path.exists(TRAIN_IMAGE_PATH):
        os.mkdir(TRAIN_IMAGE_PATH)

    ####### VALIDATE IMAGE PATH #######
    if not os.path.exists(VALIDATE_IMAGE_PATH):
        os.mkdir(VALIDATE_IMAGE_PATH)


########################## SET PREFERENCES #############################
# checking for user preferences from file

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
    print("\tNo " + TEST_IMAGE_VAR + " variable in preferences, reverting to default: " + str(DEFAULT_NUM_VAL_IMAGES))
    return DEFAULT_NUM_VAL_IMAGES


########################## SORT IMAGES #############################
# Takes all the images in the image folder and places them in test, train and validate
# Train = 90% of the images
# Test = 10% of the images
# Validate = takes user spicifed amount of files out of train
def sort_images():

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
                    shutil.move(IMAGES_PATH + filename, TEST_IMAGE_PATH)
                    shutil.move(IMAGES_PATH + xml_version, TEST_IMAGE_PATH)
                    found_label = True

            # move to train
            else:
                if path.exists(IMAGES_PATH + xml_version):
                    shutil.move(IMAGES_PATH + filename, TRAIN_IMAGE_PATH)
                    shutil.move(IMAGES_PATH + xml_version, TRAIN_IMAGE_PATH)
                    found_label = True

            # if image was found but label was not:
            if found_label == False:
                unlabelled_files.append(filename)

        # file is not a folder, image or .xml then delete it
        elif os.path.isdir(filename) and '.xml' not in filename:
            os.remove(filename)

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
    while len(valid_images) < get_valid_image_num():
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


def build_example(annotation, class_map, input_folder):
    img_path = os.path.join(input_folder, annotation['filename'])
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])


    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult_obj = []
    if 'object' in annotation:
        for obj in annotation['object']:
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


########################## PARSE XML #############################
def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


########################## GENERATE TFRECORDS #############################
def generate_tfrecords(input_folder, output_file):
    class_map = {name: idx for idx, name in enumerate(
        open(CLASSIFIER_FILE).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    writer = tf.io.TFRecordWriter(output_file)
    image_list = []
    # r=root, d=directories, f = files
    for file in os.listdir(input_folder):
        if '.jpg' in file:
            image_list.append(file) #remove .jpg suffix
    logging.info("Image list loaded: %d", len(image_list))
    for image in tqdm.tqdm(image_list):
        name = image[:len(image) - 4]
        annotation_xml = os.path.join(input_folder, name + '.xml')
        annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
        annotation = parse_xml(annotation_xml)['annotation']
        tf_example = build_example(annotation, class_map, input_folder)
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info("Done")

########################## MAIN #############################
def main():

    #check if necessary files exist
    print("\nChecking that necessary file path exist...")
    checkIfNecessaryPathsAndFilesExist()
    print("\tAll necessary files exist\n")

    # create classifiers.names
    print("Gathering classifier data...")
    classifiers = get_classifiers(IMAGES_PATH)
    create_classifier_file(classifiers)
    print("\tData successfuly classified\n")

    # sort all the images
    print("Sorting images...")
    sort_images()
    print("\n\tAll images sorted\n")

    # generate tf records
    print("Generating images and xml files into tfrecords...")
    generate_tfrecords(TRAIN_IMAGE_PATH, TRAIN_TF_RECORD_PATH)
    generate_tfrecords(TEST_IMAGE_PATH, TEST_TF_RECORD_PATH)
    print("\n\tSuccessfully generated tf records")

main()
