import csv
import cv2
import functools
import glob
import io
import json
import logging
import numpy as np
import os
import pandas as pd
import random
import shutil
import sys
import tensorflow.compat.v1 as tf
import xml.etree.ElementTree as ET

from collections import namedtuple
from distutils.version import StrictVersion
from google.protobuf import text_format
from object_detection import exporter
from object_detection.legacy import trainer
from object_detection.builders import dataset_builder
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util
from object_detection.utils import dataset_util
from os import path
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

tf.disable_v2_behavior()
os.chdir("../")
CWD_PATH = os.getcwd() + "/"

######################## STATIC_VALUES #########################################
### DIRECTORIES ###
ARCHIVED = "archived_training_sessions"
ARCHIVED_PATH = CWD_PATH + "training/" + ARCHIVED
CHECKPOINT_PATH = CWD_PATH  + "training/model.ckpt-"
DATA_PATH = "data/"
FILES_MODEL_PATH_MUST_CONTAIN = [ "checkpoint" ,
                                 "frozen_inference_graph.pb",
                                 "model.ckpt.data-00000-of-00001",
                                 "model.ckpt.index",
                                 "model.ckpt.meta"]
FROZEN_GRAPH_LOC =  CWD_PATH + "training/pre-trained_model/frozen_inference_graph.pb"
GENERATE_REC = "scripts/run_tf.py"
IMAGE_PATH = CWD_PATH + "images/"
LABEL_MAP_PATH = CWD_PATH + "data/label_map.pbtxt"
PIPELINE_PATH = CWD_PATH + "training/pre-trained_model/pipeline.config"
PRE_TRAINED_MODEL_PATH = CWD_PATH + "training/pre-trained_model"
PRE_TRAINED_CHECKPOINT_PATH = CWD_PATH + "training/pre-trained_model/model.ckpt"
SAVED_MODEL = "saved_training_checkpoint-"

TEST_CSV_PATH = CWD_PATH + "data/test_labels.csv"
TEST_FILE = CWD_PATH + "scripts/test.py"
TEST_IMAGE_PATH =  IMAGE_PATH + "test/"
TEST_TF_RECORD_PATH = CWD_PATH + "data/test.tfrecord"

TRAIN_CSV_PATH = CWD_PATH + "data/train_labels.csv"
TRAIN_IMAGE_PATH = IMAGE_PATH + "train/"
TRAIN_TF_RECORD_PATH = CWD_PATH + "data/train.tfrecord"

TRAINING_PATH = CWD_PATH + "training/"
TRAINED_MODEL_PATH = CWD_PATH +  "training/trained_model"

VALIDATE_IMAGE_PATH = IMAGE_PATH + "validation/"


### OTHER VALUES ###
ACCESS_RIGHTS = 777
CLONE_ON_CPU = False
GENERATE_REC_LABEL_MAP = "def classAsTextTo" + "ClassAsInt(classAsText):"
INPUT_SHAPE = None
INPUT_TYPE = "image_tensor"
MAX_NUM_ARCHIVED = 5
MIN_IMAGES = 50
NUM_CLONES = 1
QUOTE = '"'
STEP_COUNT = 5
VALID_IMAGE_NUM = 3

#############################################################################################################################
############################################################################################################## XML TO CSV ###
#############################################################################################################################
#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    ####### IMAGE PATH #######
    if not os.path.exists(IMAGE_PATH):
        print("ERROR: The image directory does not exist")
        exit()

    ####### TEST IMAGE PATH #######
    if not os.path.exists(TEST_IMAGE_PATH):
        os.mkdir(TEST_IMAGE_PATH)

    ####### TRAIN IMAGE PATH #######
    if not os.path.exists(TRAIN_IMAGE_PATH):
        os.mkdir(TRAIN_IMAGE_PATH)

    ####### VALIDATE IMAGE PATH #######
    if not os.path.exists(VALIDATE_IMAGE_PATH):
            os.mkdir(VALIDATE_IMAGE_PATH)

   ####### PIPELINE PATH #######
    if not os.path.exists(PIPELINE_PATH):
        print('ERROR: the pipeline.config file "' + PIPELINE_PATH + '" does not seem to exist')
        exit()

    ####### PRE_TRAINED MODEL #######
    if not os.path.exists(PRE_TRAINED_MODEL_PATH):
        print('ERROR: the model directory "' + PRE_TRAINED_MODEL_PATH + '" does not seem to exist')
        exit()

    ####### MODEL_FILES #######
    for necessaryModelFileName in FILES_MODEL_PATH_MUST_CONTAIN:
        if not os.path.exists(os.path.join(PRE_TRAINED_MODEL_PATH, necessaryModelFileName)):
            print('ERROR: the model file "' + PRE_TRAINED_MODEL_PATH + "/" + necessaryModelFileName + '" does not seem to exist')
            exit()

    ####### LABEL_MAP #######
    if not os.path.exists(LABEL_MAP_PATH):
        print('ERROR: the label map file "' + LABEL_MAP_PATH + '" does not seem to exist')
        exit()

    return True

def sort_images():
    total_images = 0
    file_count = 0
    unlabelled_files = []
    valid_images = []
    get_valid = False

    ########################################## MOVE IMAGE FILES
    for filename in os.listdir(IMAGE_PATH):
        if '.png' in filename or '.jpg' in filename or '.jpeg' in filename:
            found_label = False
            xml_version = filename.split(".")[0] + ".xml"
            total_images += 1

            # move to test
            if total_images % 10 == 0:
                shutil.move(IMAGE_PATH + filename, TEST_IMAGE_PATH)
                if path.exists(IMAGE_PATH + xml_version):
                    shutil.move(IMAGE_PATH + xml_version, TEST_IMAGE_PATH)
                    found_label = True

            # move to train
            else:
                shutil.move(IMAGE_PATH + filename, TRAIN_IMAGE_PATH)
                if path.exists(IMAGE_PATH + xml_version):
                    shutil.move(IMAGE_PATH + xml_version, TRAIN_IMAGE_PATH)
                    found_label = True

            # if image was found but label was not:
            if found_label == False:
                unlabelled_files.append(filename)

        # file is not a folder, image or .xml then delete it
        elif os.path.isdir(filename) and '.xml' not in filename:
            os.remove(filename)

    total_images = 0
    train_images = 0
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

    # total images must be greater than pre-defined count to train on
    if total_images < MIN_IMAGES:
        print("\nTensorflow needs at least " + str(MIN_IMAGES) + " images to train")
        exit()

    # if files are unlabelled, print them
    if len(unlabelled_files) != 0:
        print("The following images do not have a label:\n\t")
        print("\t" + unlabelled_files)

    # move all files in validate to train
    for file in os.listdir(VALIDATE_IMAGE_PATH):
        shutil.move(VALIDATE_IMAGE_PATH + file, TRAIN_IMAGE_PATH)

    # gather all valid images from train
    while len(valid_images) < VALID_IMAGE_NUM:
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


def xml_to_csv(path):

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

################### CONVERSION_TO_CSV #####################
def convert_to_csv():
    for directory in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), 'images/{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)
        print('\tSuccessfully converted xml to csv.')

#############################################################################################################################
############################################################################################################ UPDATE FILES ###
#############################################################################################################################
def get_classifer_info():
    class_counter = 0
    classifiers = []

    with open(TRAIN_CSV_PATH) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[3] not in classifiers:
                if row[3] != "class":
                    classifiers.append(row[3])

    return classifiers

########################## PIPELINE #############################
def update_pipeline(num_classes):
    stored_lines = []
    input_test = False
    with open(PIPELINE_PATH, "r") as f:
        for line in f.readlines():
            if 'num_classes:' in line:
                stored_lines.append("    num_classes: " + num_classes + "\n")
            elif 'label_map_path:' in line:
                stored_lines.append("\tlabel_map_path: " + QUOTE + LABEL_MAP_PATH + QUOTE + "\n")
            elif 'input_path:' in line and input_test == False:
                stored_lines.append("\t  input_path: " + QUOTE + TRAIN_TF_RECORD_PATH  + QUOTE + "\n")
                input_test = True
            elif 'input_path:' in line and input_test == True:
                stored_lines.append("\t  input_path: " + QUOTE + TEST_TF_RECORD_PATH + QUOTE + "\n")
            elif 'num_steps:' in line:
                stored_lines.append("  num_steps: " + str(STEP_COUNT) + "\n")
            elif 'fine_tune_checkpoint' in line:
                stored_lines.append("  fine_tune_checkpoint: " + QUOTE + PRE_TRAINED_CHECKPOINT_PATH + QUOTE + "\n")
            else:
                stored_lines.append(line)

    with open(PIPELINE_PATH, "w") as f:
        for line in stored_lines:
            f.write(line)
    f.close()

########################## LABEL_MAP ############################
def update_label_map(classifiers):
    stored_lines = []
    class_counter = 0
    with open(LABEL_MAP_PATH, "w") as f:
        for classification in classifiers:
            class_counter += 1
            f.write("item { \n\tid: " + str(class_counter) + "\n\tname: '" + classification + "'\n}\n")

########################## GENERATE_REC ############################
def update_record_files(classifiers):
    stored_lines = []
    class_counter = 1
    change = False
    with open(GENERATE_REC, "r") as f:
        for line in f.readlines():
            if GENERATE_REC_LABEL_MAP in line and not change:
                stored_lines.append(GENERATE_REC_LABEL_MAP + "\n\n")
                for classification in classifiers:
                    if class_counter == 1:
                        stored_lines.append("\tif classAsText == '" + classification + "':\n\t\treturn " + str(class_counter) + "\n")
                    else:
                        stored_lines.append("\telif classAsText == '" + classification + "':\n\t\treturn " + str(class_counter) + "\n")
                    class_counter += 1
                change = True

            elif change and 'else' in line:
                stored_lines.append('\telse:')
                change = False
            elif not change:
                stored_lines.append(line)

    with open(GENERATE_REC, "w") as f:
        for line in stored_lines:
            f.write(line)
    f.close()

########################## TEST_FILE #############################
    stored_lines = []
    with open(TEST_FILE, "r") as f:
        for line in f.readlines():
            if 'NUM_CLASSES =' in line:
                stored_lines.append("NUM_CLASSES = " + str(len(get_classifer_info())))
            else:
                stored_lines.append(line)

    with open(TEST_FILE, "w") as f:
        for line in stored_lines:
            f.write(line)

########################## DELETE_FILES ############################
def delete_files():
    last_checkpoint = get_checkpoint()
    for filename in os.listdir():
        if 'model.ckpt-' and 'meta' in filename:
           current = filename.split('-')[1]
           current = current.split('.')[0]
           if last_checkpoint < int(current):
               last_checkpoint = int(current)
    if last_checkpoint != 0:
        create_archive()
        archive_placement = place_archive()
        for filename in os.listdir():
            if 'model.ckpt-' + str(last_checkpoint) in filename:
                print(os.listdir())
                shutil.move(filename, archive_placement)

            elif not path.isdir(filename):
                os.remove(filename)
    else:
        for filename in os.listdir():
            if not path.isdir(filename):
                os.remove(filename)

########################## GET CHECKPOINT ############################
def get_checkpoint():
    os.chdir(TRAINING_PATH)
    last_checkpoint = 0
    for filename in os.listdir():
        if 'model.ckpt-' and 'meta' in filename:
           current = filename.split('-')[1]
           current = current.split('.')[0]
           if last_checkpoint < int(current):
               last_checkpoint = int(current)
    return last_checkpoint

########################## CREATE_ARCHIVE ############################
def create_archive():
    path_created = False
    os.chdir(TRAINING_PATH)
    for filename in os.listdir():
        if ARCHIVED in filename:
           path_created = True
    if path_created == False:
        os.mkdir(ARCHIVED, ACCESS_RIGHTS)

########################## PLACE_ARCHIVE ############################
def place_archive():
    archive_counter = 1
    oldest_save = ARCHIVED_PATH + '/' + SAVED_MODEL + '1'
    print(oldest_save)
    for file in os.listdir(ARCHIVED_PATH):
        archive_counter += 1
    if archive_counter > MAX_NUM_ARCHIVED:
        shutil.rmtree(oldest_save)
        archive_counter = 0
        os.chdir(ARCHIVED_PATH)
        for filename in os.listdir():
            archive_counter += 1
            os.rename(filename, SAVED_MODEL + str(archive_counter))
        archive_counter += 1
    os.mkdir( ARCHIVED_PATH + '/' + SAVED_MODEL + str(archive_counter))
    return  ARCHIVED_PATH + '/' + SAVED_MODEL + str(archive_counter)



#############################################################################################################################
##################################################################################################### GENERATE TF RECORDS ###
#############################################################################################################################
def generate_tfrecords():

    # write the train data .tfrecord file
    trainTfRecordFileWriteSuccessful = writeTfRecordFile(TRAIN_CSV_PATH, TRAIN_TF_RECORD_PATH, TRAIN_IMAGE_PATH)
    if trainTfRecordFileWriteSuccessful:
        print("\tSuccessfully created the training TFRecords" )

    # write the eval data .tfrecord file
    evalTfRecordFileWriteSuccessful = writeTfRecordFile(TEST_CSV_PATH, TEST_TF_RECORD_PATH, TEST_IMAGE_PATH)
    if evalTfRecordFileWriteSuccessful:
        print("\tSuccessfully created the testing TFRecords" )

##################################################################
def writeTfRecordFile(csvFileName, tfRecordFileName, imagesDir):
    # use pandas to read in the .csv file data, pandas.read_csv() returns a type DataFrame with the given param
    csvFileDataFrame = pd.read_csv(csvFileName)

    # reformat the CSV data into a format TensorFlow can work with
    csvFileDataList = reformatCsvFileData(csvFileDataFrame)

    # instantiate a TFRecordWriter for the file data
    tfRecordWriter = tf.python_io.TFRecordWriter(tfRecordFileName)

    # for each file (not each line) in the CSV file data . . .
    # (each image/.xml file pair can have more than one box, and therefore more than one line for that file in the CSV file)
    for singleFileData in csvFileDataList:
        tfExample = createTfExample(singleFileData, imagesDir)
        tfRecordWriter.write(tfExample.SerializeToString())

    tfRecordWriter.close()
    return True

#######################################################################################################################
def reformatCsvFileData(csvFileDataFrame):
    # the purpose of this function is to translate the data from one CSV file in pandas.DataFrame format
    # into a list of the named tuple below, which then can be fed into TensorFlow

    # establish the named tuple data format
    dataFormat = namedtuple('data', ['filename', 'object'])

    #  pandas.DataFrame.groupby() returns type pandas.core.groupby.DataFrameGroupBy
    csvFileDataFrameGroupBy = csvFileDataFrame.groupby('filename')

    # declare, populate, and return the list of named tuples of CSV data
    csvFileDataList = []
    for filename, x in zip(csvFileDataFrameGroupBy.groups.keys(), csvFileDataFrameGroupBy.groups):
        csvFileDataList.append(dataFormat(filename, csvFileDataFrameGroupBy.get_group(x)))
    # end for
    return csvFileDataList
# end function

#######################################################################################################################
def createTfExample(singleFileData, path):
    # use TensorFlow's GFile function to open the .jpg image matching the current box data
    with tf.gfile.GFile(os.path.join(path, '{}'.format(singleFileData.filename)), 'rb') as tensorFlowImageFile:
        tensorFlowImage = tensorFlowImageFile.read()
    # end with

    # get the image width and height via converting from a TensorFlow image to an io library BytesIO image,
    # then to a PIL Image, then breaking out the width and height
    bytesIoImage = io.BytesIO(tensorFlowImage)
    pilImage = Image.open(bytesIoImage)
    width, height = pilImage.size

    # get the file name from the file data passed in, and set the image format to .jpg
    fileName = singleFileData.filename.encode('utf8')
    imageFormat = b'jpg'

    # declare empty lists for the box x, y, mins and maxes, and the class as text and as an integer
    xMins = []
    xMaxs = []
    yMins = []
    yMaxs = []
    classesAsText = []
    classesAsInts = []

    # for each row in the current .xml file's data . . . (each row in the .xml file corresponds to one box)
    for index, row in singleFileData.object.iterrows():
        xMins.append(row['xmin'] / width)
        xMaxs.append(row['xmax'] / width)
        yMins.append(row['ymin'] / height)
        yMaxs.append(row['ymax'] / height)
        classesAsText.append(row['class'].encode('utf8'))
        classesAsInts.append(classAsTextToClassAsInt(row['class']))
    # end for

    # finally we can calculate and return the TensorFlow Example
    tfExample = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(fileName),
        'image/source_id': dataset_util.bytes_feature(fileName),
        'image/encoded': dataset_util.bytes_feature(tensorFlowImage),
        'image/format': dataset_util.bytes_feature(imageFormat),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xMins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xMaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(yMins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(yMaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classesAsText),
        'image/object/class/label': dataset_util.int64_list_feature(classesAsInts)}))

    return tfExample
# end function

#######################################################################################################################
def classAsTextToClassAsInt(classAsText):
	if classAsText == 'd4-3':
		return 1
	else:
		return 0
		
#############################################################################################################################
################################################################################################################# TESTING ###
#############################################################################################################################
def test():
    # this next comment line is necessary to avoid a false PyCharm warning
    # noinspection PyUnresolvedReferences
    if StrictVersion(tf.__version__) < StrictVersion('1.5.0'):
        raise ImportError('Please upgrade your tensorflow installation to v1.5.* or later!')
    # end if

    # load a (frozen) TensorFlow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FROZEN_GRAPH_LOC, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        # end with
    # end with

    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`,
    # we know that this corresponds to `airplane`.  Here we use internal utility functions,
    # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    max_num_classes= len(get_classifer_info())
    label_map = label_map_util.load_labelmap(LABEL_MAP_PATH)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=max_num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    imageFilePaths = []
    for imageFileName in os.listdir(VALIDATE_IMAGE_PATH):
        if imageFileName.endswith(".jpg"):
            imageFilePaths.append(VALIDATE_IMAGE_PATH + "/" + imageFileName)
        # end if
    # end for

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in imageFilePaths:

                print(image_path)

                image_np = cv2.imread(image_path)

                if image_np is None:
                    print("error reading file " + image_path)
                    continue
                # end if

                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8)
                cv2.imshow("image_np", image_np)
                cv2.waitKey(0)


#############################################################################################################################
########################################################################################################### MAIN FUNCTION ###
#############################################################################################################################
def main(self):

    print("\nChecking directories and files...")
    checkIfNecessaryPathsAndFilesExist()
    print("\tAll files and directories found\n")

    print("\nSorting images...")
    sort_images()
    print("\tSuccessfully sorted images\n")

    print("\nConverting labelled files to csv files...")
    convert_to_csv()
    print("\tSuccessfully updated files.\n")

    print("\nUpdating classifers and config files...")
    classifiers = get_classifer_info()
    num_classes = str(len(classifiers))
    update_pipeline(num_classes)
    update_label_map(classifiers)
    update_record_files(classifiers)
    delete_files()
    print("\tSuccessfully updated files")

    print("\nConverting csv files to tensorflow records...")
    generate_tfrecords()
    print("\tSuccessfully converted csv files into tensorflow records\n")



    print("\tSuccessfully updated files.\n")

    print("\nBeing Training . . .")

#############################################################################################################################
################################################################################################################ TRAINING ###
#############################################################################################################################
    # show info to std out during the training process
    tf.logging.set_verbosity(tf.logging.INFO)

    configs = config_util.get_configs_from_pipeline_file(PIPELINE_PATH)
    tf.gfile.Copy(PIPELINE_PATH, os.path.join(TRAINING_PATH, 'pipeline.config'), overwrite=True)

    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    model_fn = functools.partial(model_builder.build, model_config=model_config, is_training=True)

    # ToDo: this nested function seems odd, factor this out eventually ??
    # nested function
    def get_next(config):
        return dataset_builder.make_initializable_iterator(dataset_builder.build(config)).get_next()
    # end nested function

    create_input_dict_fn = functools.partial(get_next, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    # parameters for a single worker
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    if cluster_data and 'worker' in cluster_data:
        # number of total worker replicas include "worker"s and the "master".
        worker_replicas = len(cluster_data['worker']) + 1
    # end if

    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])
    # end if

    if worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('At least 1 ps task is needed for distributed training.')
    # end if

    if worker_replicas >= 1 and ps_tasks > 0:
        # set up distributed training
        server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc', job_name=task_info.type, task_index=task_info.index)
        if task_info.type == 'ps':
            server.join()
            return
        # end if

        worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
        task = task_info.index
        is_chief = (task_info.type == 'master')
        master = server.target
    # end if

    trainer.train(create_input_dict_fn, model_fn, train_config, master, task, NUM_CLONES, worker_replicas,
                  CLONE_ON_CPU, ps_tasks, worker_job_name, is_chief, TRAINING_PATH)

#############################################################################################################################
################################################################################################## EXPORT INFERENCE GRAPH ###
#############################################################################################################################
    checkpoint = get_checkpoint()

    if path.exists(TRAINED_MODEL_PATH):
        shutil.rmtree(TRAINED_MODEL_PATH)
    os.mkdir(TRAINED_MODEL_PATH, ACCESS_RIGHTS)
    print("Exporting inference graph...")

    print("calling TrainEvalPipelineConfig() . . .")
    trainEvalPipelineConfig = pipeline_pb2.TrainEvalPipelineConfig()

    print("checking and merging " + os.path.basename(PIPELINE_PATH) + " into trainEvalPipelineConfig . . .")
    with tf.gfile.GFile(PIPELINE_PATH, 'r') as f:
        text_format.Merge(f.read(), trainEvalPipelineConfig)
    # end with

    print("calculating input shape . . .")
    if INPUT_SHAPE:
        input_shape = [ int(dim) if dim != '-1' else None for dim in INPUT_SHAPE.split(',') ]
    else:
        input_shape = None
    # end if

    print("calling export_inference_graph() . . .")
    exporter.export_inference_graph(INPUT_TYPE, trainEvalPipelineConfig, CHECKPOINT_PATH + str(checkpoint), TRAINED_MODEL_PATH, INPUT_SHAPE)

    print("Successfully exported Inference graph")

    print("\nStarting testing. . .")

    test()

# end main
if __name__ == '__main__':
    tf.app.run()
