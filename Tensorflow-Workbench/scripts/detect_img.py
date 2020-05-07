import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

def from_workbench(path):
    keyword = "Workbench"
    path = path.replace("//", "/")
    try:
        if keyword in path:
            return "." + path.split(keyword)[1]
        else:
            return path
    except:
        return str(path)


def run_detect(classes, weights, tiny, size, image, output, num_classes):
    tfrecord = None
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #TODO figures this out is its important as this causing an error due to the physical device being initialized already
    #if len(physical_devices) > 0:
        #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if tiny:
        yolo = YoloV3Tiny(classes=num_classes)
    else:
        yolo = YoloV3(classes=num_classes)
    yolo.load_weights(weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(classes).readlines()]
    logging.info('classes loaded')

    if tfrecord:
        dataset = load_tfrecord_dataset(
            tfrecord, classes, size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)

    t1 = time.time()
    boxes, scores, classesArr, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    no_classifier = True
    print("\n\tTesting on image: " + from_workbench(image) + "\n")
    for i in range(nums[0]):
        object_name = class_names[int(classesArr[0][i])]
        object_acc = round(100 * np.array(scores[0][i]), 2)
        object_loc = np.array(boxes[0][i])
        print("\t\tObject " + str(i + 1) + ": " + str(object_name) + " with a " + str(object_acc) + "% accuracy\n")
        no_classifier = False
        logging.info('\t{}, {}, {}'.format(class_names[int(classesArr[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))
    if no_classifier:
        print("\t\tNo objects found in image\n")
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classesArr, nums), class_names)
    cv2.imwrite(output, img)
    logging.info('output saved to: {}'.format(output))
