from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import sys
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.utils import load_darknet_weights

def run_weight_convert(weights, output, tiny, num_classes):
    output = output.replace("//", "/")

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
