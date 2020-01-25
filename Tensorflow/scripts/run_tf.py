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

######################## STATIC_VALUES #########################################
### DIRECTORIES ###
ARCHIVED = "archived_training_sessions"
ARCHIVED_DIR = "../training/" + ARCHIVED
CHECKPOINT_PATH = os.getcwd() +  "/../training/model.ckpt-"
CWD_PATH = os.getcwd()
DATA_DIR = "../data/"
FILES_MODEL_DIR_MUST_CONTAIN = [ "checkpoint" ,
                                 "frozen_inference_graph.pb",
                                 "model.ckpt.data-00000-of-00001",
                                 "model.ckpt.index",
                                 "model.ckpt.meta"]
FROZEN_GRAPH_LOC =  os.getcwd() + "/../training/pre-trained_model/frozen_inference_graph.pb"
GENERATE_REC = "run_tf.py"
IMAGE_DIR = "../images/"
LABEL_MAP_PATH = "../data/label_map.pbtxt"
PRE_TRAINED_MODEL_DIR = "../training/pre-trained_model/"
PIPELINE_PATH = PRE_TRAINED_MODEL_DIR + "pipeline.config"
SAVED_MODEL = "saved_training_checkpoint-"

TEST_CSV_PATH = DATA_DIR + 'test_labels.csv'
TEST_FILE_PATH =  "test.py"
TEST_IMAGE_PATH = IMAGE_DIR + "test"
TEST_TF_RECORD_PATH = DATA_DIR + 'test.tfrecord'

TRAIN_CSV_PATH = DATA_DIR + 'train_labels.csv'
TRAIN_IMAGE_PATH = IMAGE_DIR + "train"
TRAIN_TF_RECORD_PATH = DATA_DIR + 'train.tfrecord'

TRAINING_DIR = "../training/"
TRAINED_MODEL_DIR = TRAINING_DIR + "trained_model"

VALIDATE_IMAGE_DIR = IMAGE_DIR + "validation"


### OTHER VALUES ###
ACCESS_RIGHTS = 777
CLONE_ON_CPU = False
GENERATE_REC_LABEL_MAP = "def classAsTextTo" + "ClassAsInt(classAsText):"
INPUT_SHAPE = None
INPUT_TYPE = "image_tensor"
MAX_NUM_ARCHIVED = 5
NUM_CLONES = 1
QUOTE = '"'
STEP_COUNT = 1000

#############################################################################################################################
############################################################################################################## XML TO CSV ###
#############################################################################################################################
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
        image_path = os.path.join(os.getcwd(), '../images/{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('../data/{}_labels.csv'.format(directory), index=None)
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
                stored_lines.append("\tnum_classes: " + num_classes + "\n")
            elif 'label_map_path:' in line:
                stored_lines.append("\tlabel_map_path: " + QUOTE + LABEL_MAP_PATH + QUOTE + "\n")
            elif 'input_path:' in line and input_test == False:
                stored_lines.append("\tinput_path: " + QUOTE + TRAIN_TF_RECORD_PATH  + QUOTE + "\n")
                input_test = True
            elif 'input_path:' in line and input_test == True:
                stored_lines.append("\tinput_path: " + QUOTE + TEST_TF_RECORD_PATH + QUOTE + "\n")
            elif 'num_steps:' in line:
                stored_lines.append("  num_steps: " + str(STEP_COUNT) + "\n")
            else:
                stored_lines.append(line)

    with open(PIPELINE_PATH, "w") as f:
        for line in stored_lines:
            f.write(line)

########################## LABEL_MAP ############################
def update_label_map(classifiers):
    stored_lines = []
    class_counter = 0
    with open(LABEL_MAP_PATH, "w") as f:
        for classification in classifiers:
            class_counter += 1
            f.write("item { \n\tid: " + str(class_counter) + "\n\tname: '" + classification + "'\n}\n\n")

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
                        stored_lines.append("\tif classAsText == '" + classification + "':\n\t\t return " + str(class_counter) + "\n")
                    else:
                        stored_lines.append("\telif classAsText == '" + classification + "':\n\t\t return " + str(class_counter) + "\n")
                    class_counter += 1
                change = True
            elif change and 'else' in line:
                stored_lines.append('\telse:\n')
                change = False
            elif not change:
                stored_lines.append(line)

    with open(GENERATE_REC, "w") as f:
        for line in stored_lines:
            f.write(line)

########################## DELETE_FILES ############################
def delete_files():
    last_checkpoint = 0
    os.chdir(TRAINING_DIR)
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
    return(last_checkpoint)


########################## CREATE_ARCHIVE ############################
def create_archive():
    path_created = False
    os.chdir(TRAINING_DIR)
    for filename in os.listdir():
        if ARCHIVED in filename:
           path_created = True
    if path_created == False:
        os.mkdir(ARCHIVED, ACCESS_RIGHTS)

########################## PLACE_ARCHIVE ############################
def place_archive():
    archive_counter = 1
    oldest_save = ARCHIVED_DIR + '/' + SAVED_MODEL + '1'
    print(oldest_save)
    for file in os.listdir(ARCHIVED_DIR):
        archive_counter += 1
    if archive_counter > MAX_NUM_ARCHIVED:
        shutil.rmtree(oldest_save)
        archive_counter = 0
        os.chdir(ARCHIVED_DIR)
        for filename in os.listdir():
            archive_counter += 1
            os.rename(filename, SAVED_MODEL + str(archive_counter))
        archive_counter += 1
        os.chdir('..')
    os.mkdir( ARCHIVED_DIR + '/' + SAVED_MODEL + str(archive_counter))
    return  ARCHIVED_DIR + '/' + SAVED_MODEL + str(archive_counter)



#############################################################################################################################
##################################################################################################### GENERATE TF RECORDS ###
#############################################################################################################################

def generate_tfrecords():
    if not checkIfNecessaryPathsAndFilesExist():
        return
    # end if

    # write the train data .tfrecord file
    trainTfRecordFileWriteSuccessful = writeTfRecordFile(TRAIN_CSV_PATH, TRAIN_TF_RECORD_PATH, TRAIN_IMAGE_PATH)
    if trainTfRecordFileWriteSuccessful:
        print("\tSuccessfully created the training TFRecords" )
    # end if

    # write the eval data .tfrecord file
    evalTfRecordFileWriteSuccessful = writeTfRecordFile(TEST_CSV_PATH, TEST_TF_RECORD_PATH, TEST_IMAGE_PATH)
    if evalTfRecordFileWriteSuccessful:
        print("\tSuccessfully created the testing TFRecords" )
    # end if

# end main

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
    # end for
    tfRecordWriter.close()
    return True        # return True to indicate success
# end function

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(TRAIN_CSV_PATH):
        print('ERROR: TRAIN_CSV_FILE "' + TRAIN_CSV_PATH + '" does not seem to exist')
        return False

    if not os.path.exists(TRAIN_IMAGE_PATH):
        print('ERROR: TRAIN_IMAGES_DIR "' + TRAIN_IMAGE_PATH + '" does not seem to exist')
        return False

    if not os.path.exists(TEST_CSV_PATH):
        print('ERROR: TEST_CSV_FILE "' + TEST_CSV_PATH + '" does not seem to exist')
        return False

    if not os.path.exists(TEST_IMAGE_PATH):
        print('ERROR: TEST_IMAGES_DIR "' + TEST_IMAGE_PATH + '" does not seem to exist')
        return False

    if not os.path.exists(PIPELINE_PATH):
        print('ERROR: the big (pipeline).config file "' + PIPELINE_PATH + '" does not seem to exist')
        return False

    if not os.path.exists(PRE_TRAINED_MODEL_DIR):
        print('ERROR: the model directory "' + PRE_TRAINED_MODEL_DIR + '" does not seem to exist')
        print(missingModelMessage)
        return False

    for necessaryModelFileName in FILES_MODEL_DIR_MUST_CONTAIN:
        if not os.path.exists(os.path.join(PRE_TRAINED_MODEL_DIR, necessaryModelFileName)):
            print('ERROR: the model file "' + PRE_TRAINED_MODEL_DIR + "/" + necessaryModelFileName + '" does not seem to exist')
            print(missingModelMessage)
            return False

    if not os.path.exists(TRAINING_DIR):
        print('ERROR: TRAINING_DIR "' + TRAINING_DIR + '" does not seem to exist')
        return False

    if not os.path.exists(PIPELINE_PATH):
        print('ERROR: PIPELINE_CONFIG_LOC "' + PIPELINE_PATH + '" does not seem to exist')
        return False

    trainedCkptPrefixPath, filePrefix = os.path.split(CHECKPOINT_PATH )

    if not os.path.exists(trainedCkptPrefixPath):
        print('ERROR: directory "' + trainedCkptPrefixPath + '" does not seem to exist')
        print('was the training completed successfully?')
        return False

    numFilesThatStartWithPrefix = 0

    for fileName in os.listdir(trainedCkptPrefixPath):
        if fileName.startswith(filePrefix):
            numFilesThatStartWithPrefix += 1

    if not os.path.exists(VALIDATE_IMAGE_DIR):
        print('ERROR: VALIDATE_IMAGE_DIR "' + VALIDATE_IMAGE_DIR + '" does not seem to exist')
        return False

    if not os.path.exists(LABEL_MAP_PATH):
        print('ERROR: the label map file "' + LABEL_MAP_PATH + '" does not seem to exist')
        return False

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

	if classAsText == 'd4-1':
		 return 1
	elif classAsText == 'd4-2':
		 return 2
	elif classAsText == 'd4-3':
		 return 3
	elif classAsText == 'd4-4':
		 return 4
	elif classAsText == 'd6-1':
		 return 5
	elif classAsText == 'd6-2':
		 return 6
	elif classAsText == 'd6-3':
		 return 7
	elif classAsText == 'd6-4':
		 return 8
	elif classAsText == 'd6-5':
		 return 9
	elif classAsText == 'd6-6':
		 return 10
	elif classAsText == 'd8-1':
		 return 11
	elif classAsText == 'd8-2':
		 return 12
	elif classAsText == 'd8-3':
		 return 13
	elif classAsText == 'd8-4':
		 return 14
	elif classAsText == 'd8-5':
		 return 15
	elif classAsText == 'd8-6':
		 return 16
	elif classAsText == 'd8-7':
		 return 17
	elif classAsText == 'd8-8':
		 return 18
	else:
         return -1

#############################################################################################################################
################################################################################################################# TESTING ###
#############################################################################################################################
def test():
    if not checkIfNecessaryPathsAndFilesExist():
        print("No images in Tensorflow/images/validation:")
        print("\tplace photos in this director if fyou want to validate the inference graph")
        return
    # end if

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
    label_map = label_map_util.load_labelmap(LABEL_MAP_PATH)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes= len(get_classifer_info()),
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    imageFilePaths = []
    for imageFileName in os.listdir(VALIDATE_IMAGE_DIR):
        if imageFileName.endswith(".jpg"):
            imageFilePaths.append(VALIDATE_IMAGE_DIR + "/" + imageFileName)
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

    print("Converting labelled files to csv files...")
    convert_to_csv()

    print('\n')
    print("Converting csv files to tensorflow records...")
    generate_tfrecords()

    print('\n')
    print("Updating classifers and config files...")
    classifiers = get_classifer_info()
    num_classes = str(len(classifiers))
    update_pipeline(num_classes)
    update_label_map(classifiers)
    update_record_files(classifiers)
    checkpoint = str(delete_files())

    print("\tSuccessfully updated files.")

    print("\n\nBeing Training . . .")

#############################################################################################################################
################################################################################################################ TRAINING ###
#############################################################################################################################
    # show info to std out during the training process
    tf.logging.set_verbosity(tf.logging.INFO)

    if not checkIfNecessaryPathsAndFilesExist():
        return
    # end if

    configs = config_util.get_configs_from_pipeline_file(PIPELINE_PATH)
    tf.gfile.Copy(PIPELINE_PATH, os.path.join(TRAINING_DIR, 'pipeline.config'), overwrite=True)

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
                  CLONE_ON_CPU, ps_tasks, worker_job_name, is_chief, TRAINING_DIR)

#############################################################################################################################
################################################################################################## EXPORT INFERENCE GRAPH ###
#############################################################################################################################
    if path.exists(TRAINED_MODEL_DIR):
        shutil.rmtree(TRAINED_MODEL_DIR)
    os.mkdir(TRAINED_MODEL_DIR, ACCESS_RIGHTS)
    print("Exporting inference graph...")

    if not checkIfNecessaryPathsAndFilesExist():
        return
    # end if

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
    exporter.export_inference_graph(INPUT_TYPE, trainEvalPipelineConfig, CHECKPOINT_PATH + str(checkpoint), TRAINED_MODEL_DIR, INPUT_SHAPE)

    print("Successfully exported Inference graph")

    print("\nStarting testing. . .")

    test()

# end main
if __name__ == '__main__':
    tf.app.run()
