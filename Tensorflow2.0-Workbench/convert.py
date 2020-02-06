from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.utils import load_darknet_weights

flags.DEFINE_string('weights', './data/yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov3.tf', 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def run_weight_convert(weights, output, tiny, num_classes):
    if tiny:
        yolo = YoloV3Tiny(classes=num_classes)
    else:
        yolo = YoloV3(classes=num_classes)
    yolo.summary()
    logging.info('model created')

    load_darknet_weights(yolo, weights, tiny)
    logging.info('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    outputSanity = yolo(img)
    logging.info('sanity check passed')

    yolo.save_weights(output)
    logging.info('weights saved')


def main(_argv):
    run_weight_convert(FLAGS.weights, FLAGS.output, FLAGS.tiny, FLAGS.num_classes)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
