from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import sys
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.utils import load_darknet_weights

########################## RUN_WEIGHT_CONVERT #############################
# Description: converts weights file to a chekpoint
# Parameters: weights - String - file that contains pre-trained weights_path
#             output - String - location of wehre checkpoint is saved
#             tiny - Boolean - value if using tiny weights
#             num_classes - number of classes the weights file was trained on
# Return: Nothing
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
