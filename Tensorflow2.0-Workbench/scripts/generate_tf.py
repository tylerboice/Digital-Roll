import hashlib
import lxml.etree
import os
import re
import tensorflow as tf
import time
import tqdm

from absl import app, flags, logging
from absl.flags import FLAGS

from scripts import defaults


# DATA SET PATHS
IMAGES_PATH = defaults.IMAGES_PATH
TRAIN_IMAGE_PATH = defaults.TRAIN_IMAGE_PATH
TEST_IMAGE_PATH = defaults.TEST_IMAGE_PATH
VALIDATE_IMAGE_PATH = defaults.VALIDATE_IMAGE_PATH


CLASSIFIER_FILE = defaults.CLASSIFIER_FILE
PREFERENCES_PATH = defaults.PREFERENCES_PATH
TRAIN_TF_RECORD_PATH = defaults.TRAIN_TF_RECORD_PATH
TEST_TF_RECORD_PATH = defaults.TEST_TF_RECORD_PATH


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
