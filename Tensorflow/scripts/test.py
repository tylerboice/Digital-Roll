# original file by Google:
# https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py
import cv2
import logging
import numpy as np
import os
import shutil
import sys
import tensorflow.compat.v1 as tf

from distutils.version import StrictVersion
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2
from os import path
from utils import label_map_util
from utils import visualization_utils as vis_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True
tf.disable_v2_behavior()


# module-level variables ##############################################################################################

# INPUT_TYPE can be "image_tensor", "encoded_image_string_tensor", or "tf_example"
INPUT_TYPE = "image_tensor"
ACCESS_RIGHTS = 777
INPUT_SHAPE = None

### Directories ###
TRAINED_MODEL_DIR = os.getcwd() + "/../training/trained_model"
PIPELINE_CONFIG_LOC =  os.getcwd() + "/../training/pre-trained_model/" + "pipeline.config"
TRAINED_CHECKPOINT_PREFIX_LOC = os.getcwd() +  "/../training/model.ckpt-"
TEST_IMAGE_DIR = os.getcwd() +  "/../images/validation"
FROZEN_INFERENCE_GRAPH_LOC = os.getcwd() + "/../training/pre-trained_model/frozen_inference_graph.pb"
LABELS_LOC = os.getcwd() + "/../data/" + "label_map.pbtxt"
NUM_CLASSES = 18

last_checkpoint = 0

for file in os.listdir('../training'):
    if 'model.ckpt-' and 'meta' in file:
       current = file.split('-')[1]
       current = current.split('.')[0]
       if last_checkpoint < int(current):
           last_checkpoint = int(current)

if last_checkpoint == 0:
    print("\n\n\nNo checkpoint found")
    exit()

TRAINED_CHECKPOINT_PREFIX_LOC = TRAINED_CHECKPOINT_PREFIX_LOC + str(last_checkpoint)
# the output directory to place the inference graph data, note that it's ok if this directory does not already exist
# because the call to export_inference_graph() below will create this directory if it does not exist already
OUTPUT_DIR = os.getcwd() + "/../training/trained_model/"

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
        with tf.gfile.GFile(FROZEN_INFERENCE_GRAPH_LOC, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        # end with
    # end with

    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(LABELS_LOC)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    imageFilePaths = []
    for imageFileName in os.listdir(TEST_IMAGE_DIR):
        if imageFileName.endswith(".jpg"):
            imageFilePaths.append(TEST_IMAGE_DIR + "/" + imageFileName)
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
#######################################################################################################################
def main(_):
    if path.exists(TRAINED_MODEL_DIR):
        shutil.rmtree(TRAINED_MODEL_DIR)
    os.mkdir(TRAINED_MODEL_DIR, ACCESS_RIGHTS)
    print("Exporting inference graph...")

    if not checkIfNecessaryPathsAndFilesExist():
        return
    # end if

    print("calling TrainEvalPipelineConfig() . . .")
    trainEvalPipelineConfig = pipeline_pb2.TrainEvalPipelineConfig()

    print("checking and merging " + os.path.basename(PIPELINE_CONFIG_LOC) + " into trainEvalPipelineConfig . . .")
    with tf.gfile.GFile(PIPELINE_CONFIG_LOC, 'r') as f:
        text_format.Merge(f.read(), trainEvalPipelineConfig)
    # end with

    print("calculating input shape . . .")
    if INPUT_SHAPE:
        input_shape = [ int(dim) if dim != '-1' else None for dim in INPUT_SHAPE.split(',') ]
    else:
        input_shape = None
    # end if

    print("calling export_inference_graph() . . .")
    exporter.export_inference_graph(INPUT_TYPE, trainEvalPipelineConfig, TRAINED_CHECKPOINT_PREFIX_LOC, OUTPUT_DIR, INPUT_SHAPE)

    print("Successfully exported Inference graph")

    print("\nStarting testing. . .")
    test()

# end main

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(PIPELINE_CONFIG_LOC):
        print('ERROR: PIPELINE_CONFIG_LOC "' + PIPELINE_CONFIG_LOC + '" does not seem to exist')
        return False
    # end if

    # TRAINED_CHECKPOINT_PREFIX_LOC is a special case because there is no actual file with this name.
    # i.e. if TRAINED_CHECKPOINT_PREFIX_LOC is:
    # "C:\Users\cdahms\Documents\TensorFlow_Tut_3_Object_Detection_Walk-through\training_data\training_data\model.ckpt-500"
    # this exact file does not exist, but there should be 3 files including this name, which would be:
    # "model.ckpt-500.data-00000-of-00001"
    # "model.ckpt-500.index"
    # "model.ckpt-500.meta"
    # therefore it's necessary to verify that the stated directory exists and then check if there are at least three files
    # in the stated directory that START with the stated name

    # break out the directory location and the file prefix
    trainedCkptPrefixPath, filePrefix = os.path.split(TRAINED_CHECKPOINT_PREFIX_LOC)

    # return false if the directory does not exist
    if not os.path.exists(trainedCkptPrefixPath):
        print('ERROR: directory "' + trainedCkptPrefixPath + '" does not seem to exist')
        print('was the training completed successfully?')
        return False
    # end if

    # count how many files in the stated directory start with the stated prefix
    numFilesThatStartWithPrefix = 0
    for fileName in os.listdir(trainedCkptPrefixPath):
        if fileName.startswith(filePrefix):
            numFilesThatStartWithPrefix += 1
        # end if
    # end if

    # if less than 3 files start with the stated prefix, return false
    if numFilesThatStartWithPrefix < 3:
        print('ERROR: 3 files statring with "' + filePrefix + '" do not seem to be present in the directory "' + trainedCkptPrefixPath + '"')
        print('was the training completed successfully?')
    # end if

    if not os.path.exists(TEST_IMAGE_DIR):
        print('ERROR: TEST_IMAGE_DIR "' + TEST_IMAGE_DIR + '" does not seem to exist')
        return False
    # end if

    # ToDo: check here that the test image directory contains at least one image

    if not os.path.exists(FROZEN_INFERENCE_GRAPH_LOC):
        print('ERROR: FROZEN_INFERENCE_GRAPH_LOC "' + FROZEN_INFERENCE_GRAPH_LOC + '" does not seem to exist')
        print('was the inference graph exported successfully?')
        return False
    # end if

    if not os.path.exists(LABELS_LOC):
        print('ERROR: the label map file "' + LABELS_LOC + '" does not seem to exist')
        return False
    # end if

    # if we get here the necessary directories and files are present, so return True
    return True
# end function

#######################################################################################################################
if __name__ == '__main__':
    tf.app.run()
