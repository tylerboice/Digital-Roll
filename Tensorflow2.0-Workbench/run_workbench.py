
import csv
import cv2
import hashlib
import glob
import lxml.etree
import numpy as np
import os
import pandas as pd
import random
import re
import shutil
import tensorflow as tf
import time
import tqdm

from absl import app, flags, logging
from absl.flags import FLAGS
from os import path
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

CWD_PATH = os.getcwd() + "/"

# IMAGE SET PATHS
IMAGES_PATH = CWD_PATH + "images/"
TRAIN_IMAGE_PATH = IMAGES_PATH + "train/"
TEST_IMAGE_PATH = IMAGES_PATH + "test/"
VALIDATE_IMAGE_PATH = IMAGES_PATH + "validate/"

# DATA PATHS
DATA_PATH = CWD_PATH + "data" + "/"

CLASSIFIER_FILE = DATA_PATH + "classifier.names"
PREFERENCES_PATH = CWD_PATH + "preferences.txt"
TRAIN_TF_RECORD_PATH = DATA_PATH + "train.tfrecord"
TEST_TF_RECORD_PATH = DATA_PATH + "test.tfrecord"

# HARD CODED VALUES
MIN_IMAGES = 50
YOLO_TRAINED_NUM_CLASSES = 80

# Default values
DEFAULT_BATCH_SIZE = 12
DEFAULT_EPOCH_NUM = 10
DEFAULT_IMAGE_SIZE = 416
DEFAULT_LEARN_RATE = 1e-3
DEFAULT_NUM_VAL_IMAGES = 3
DEFAULT_WEIGHT_PATH = DATA_PATH + "yolov3.weights"
DEFAULT_WEIGHT_TYPE = False #False = yolov3 True = tiny

# PREFERENCE VARIABLE NAMES
BATCH_SIZE_VAR = "batch_size:"
EPOCH_NUM_VAR = "num_epochs:"
IMAGE_SIZE_VAR = "image_size:"
LEARN_RATE_VAR = "learn_rate:"
TEST_IMAGE_VAR = 'validate_images:'
WEIGHTS_PATH_VAR = "weights_path:"
WEIGHTS_TYPE_VAR = "weights_type:"

########################## Checking FOR FILES #############################
# checks if all necessary files exist
def checkIfNecessaryPathsAndFilesExist():

    ####### IMAGE PATH #######
    if not os.path.exists(IMAGES_PATH):
        print_error(IMAGE_PATH)

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
                        return True
                    elif "yolo-tiny"in weight_type :
                        return False
    print("\tNo " + WEIGHTS_TYPE_VAR + " variable in preferences or path does not exist, reverting to default")
    return DEFAULT_WEIGHT_TYPE

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
def generate_tfrecods(input_folder, output_file):
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

############################## MAIN ##########################
def main(_argv):
    # get preferences/data
    # batch_size
    flags.DEFINE_integer('batch_size', get_batch_size(PREFERENCES_PATH), 'batch size')
    # ephoch num
    flags.DEFINE_integer('epochs', get_epoch_num(PREFERENCES_PATH), 'number of epochs')
    # image_size
    flags.DEFINE_integer('size', get_image_size(PREFERENCES_PATH), 'image size')
    # learn rate
    flags.DEFINE_float('learning_rate', get_learn_rate(PREFERENCES_PATH), 'learning rate')
    # num_classes
    flags.DEFINE_integer('num_classes', get_num_classes(CLASSIFIER_FILE), 'number of classes in the model')
    # weights_path
    flags.DEFINE_string('weights', get_weight_path(PREFERENCES_PATH), 'path to weights file')
    # tiny_weights
    flags.DEFINE_boolean('tiny', is_tiny_weight(PREFERENCES_PATH), 'yolov3 or yolov3-tiny')

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
    generate_tfrecods(TRAIN_IMAGE_PATH, TRAIN_TF_RECORD_PATH)
    generate_tfrecods(TEST_IMAGE_PATH, TEST_TF_RECORD_PATH)
    print("\n\tSuccessfully generated tf records")

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if tiny_weights:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    train_dataset = dataset.load_fake_dataset()
    if TRAIN_TF_RECORD_PATH:
        train_dataset = dataset.load_tfrecord_dataset(
            TRAIN_TF_RECORD_PATH, FLAGS.classes, FLAGS.size)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_fake_dataset()
    if TEST_TF_RECORD_PATH:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    # Configure the model for transfer learning
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    elif FLAGS.transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes

        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        else:
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))

        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                    freeze_all(l)

    else:
        # All other transfer require matching classes
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info("{}_val_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(
                'checkpoints/yolov3_train_{}.tf'.format(epoch))
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(FLAGS.mode == 'eager_fit'))

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]

        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)


    if __name__ == '__main__':
        try:
            app.run(main)
        except SystemExit:
            pass
