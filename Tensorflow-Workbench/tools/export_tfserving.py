import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest

flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('output', './serving/yolov3/1', 'path to saved_model')
flags.DEFINE_string('classes', './data/dice.names', 'path to classes file')
flags.DEFINE_string('image', './data/dice.png', 'path to input image')
flags.DEFINE_integer('num_classes', 4, 'number of classes in the model')

def run_export_tfserving(weights, tiny, output, classes, image, num_classes):
    if tiny:
        yolo = YoloV3Tiny(classes= num_classes)
    else:
        yolo = YoloV3(classes= num_classes)

    yolo.load_weights(weights)
    logging.info('weights loaded')

    tf.saved_model.save(yolo, output)
    logging.info("model saved to: {}".format(output))

    model = tf.saved_model.load(output)
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    logging.info(infer.structured_outputs)

    class_names = [c.strip() for c in open(classes).readlines()]
    logging.info('classes loaded')

    img = tf.image.decode_image(open(image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, 416)

    t1 = time.time()
    outputs = infer(img)
    boxes, scores, classes, nums = outputs["yolo_nms"], outputs[
        "yolo_nms_1"], outputs["yolo_nms_2"], outputs["yolo_nms_3"]
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           scores[0][i].numpy(),
                                           boxes[0][i].numpy()))


def main(_argv):
    run_export_tfserving(FLAGS.weights, FLAGS.tiny, FLAGS.output,
                         FLAGS.classes, FLAGS.image, FLAGS.num_classes)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
