import os
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
from tensorflow.lite.python.interpreter import load_delegate
import platform
from os import path

os.chdir("..")

CWD_PATH = os.getcwd().replace("\\", "/") + "/"

flags.DEFINE_string('model', CWD_PATH + 'current_session/',
                    'path to model file')
flags.DEFINE_string('output', CWD_PATH + 'current_session/yolov3.tflite',
                    'path to saved_model')
flags.DEFINE_string('classes', CWD_PATH + 'data/classifier.names', 'path to classes file')
flags.DEFINE_string('image', CWD_PATH + 'data/dice.jpg', 'path to input image')
flags.DEFINE_integer('num_classes', 18, 'number of classes in the model')
flags.DEFINE_integer('size', 224, 'image size')

# attempt at using delegates, failed due to bad support and system dependency
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def main(_argv):
    model = tf.saved_model.load(FLAGS.model)
    keras_model = tf.keras.models.load_model(FLAGS.model)
    print("Model Loaded")

    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    # Methods of changing output shape attempted:
        # tf.expand_dims(concrete_func.outputs[0], 0) --x> TypeError, an op outside of the function building...
        # concrete_func = tf.reshape(concrete_func.outputs, [1, 1000, 4]) --x> TypeError, an op outside of the function building...
        # tf.expand_dims(concrete_func.outputs[0], 0) --x> TypeError, an op outside of the function building...
        # also added experimental_run_tf_function=False to the .comile call in train_workbench, did not fix above errors
        # tried to give a delegate to the Interpreter via giving expermental_delegates = load_delegates('edgetpu.dll')
                # as an argument, failed due to delegates class being faulty from tf.lite, found an online suggestion to
                # try import tf_runtime as tflite, but fald to properly download the package

    # converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.model)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.experimental_new_converter = True
    converter.allow_custom_ops = False # TFLite does not support custom operations, must figure out a way to remove nms
        # when set to true the converter will produce a good output shape, but have a custom function which is its own issue
    tflite_model = converter.convert()
    open(FLAGS.output, "wb").write(tflite_model)

    logging.info("model saved to: {}".format(FLAGS.output))
    interpreter = tf.lite.Interpreter(model_path=FLAGS.output)
    interpreter.allocate_tensors()
    logging.info('tflite model loaded')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, 224)

    t1 = time.time()
    outputs = interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(output_data)

if __name__ == '__main__':
    app.run(main)
