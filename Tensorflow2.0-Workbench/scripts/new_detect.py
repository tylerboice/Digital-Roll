import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from scripts import default

# input image
flags.DEFINE_string('image_output', get_valid_test_image(), 'path to output')

def run_detect(classes_in, weights_in, tiny_in,
               size_in, image_in, tfrecord_in,
               output_in, num_classes_in):
    classes = classes_in
    weights = weights_in
    tiny = tiny_in
    size = size_in
    image = image_in
    tfrecord = tfrecord_in
    output = output_in
    num_classes = num_classes_in
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
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
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classesArr[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classesArr, nums), class_names)
    cv2.imwrite(output, img)
    logging.info('output saved to: {}'.format(output))



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
