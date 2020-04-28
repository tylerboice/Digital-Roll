import time
from absl import app, flags, logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import freeze_all

# Funcion: run_weight_convert
# Description: converts weights file to a chekpoint
# Parameters: weights - String - file that contains pre-trained
#             tiny - Boolean - value if using tiny weights
#             output - String - location of wehre checkpoint is saved
#             classes - String - path to the classifier.names files
#             image - Int - image size
#             num_classes - number of classes the model was trained on
# Return: Nothing
def run_export_tfserving(weights, tiny, output, classes, image, num_classes):


    if tiny:
        yolo = YoloV3Tiny(classes= num_classes)
    else:
        yolo = YoloV3(classes= num_classes)

    print("\n\tSaving using the weights: " + weights + "\n")
    yolo.load_weights(weights)
    logging.info('weights loaded')

    #freeze_all(yolo)

    tf.saved_model.save(yolo, output)
    logging.info("model saved to: {}".format(output))

    model = tf.saved_model.load(output)
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    logging.info(infer.structured_outputs)

    # Cannot make a .h5 without calling .expectPartial() when loading weights
    # tf.keras.models.save_model(model, output + "keras_model.h5", save_format='tf')
    # logging.info("keras version saved to: {}".format(output))


    class_names = [c.strip() for c in open(classes).readlines()]
    logging.info('classes loaded')

    img = tf.image.decode_image(open(image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, 224)

    t1 = time.time()
    outputs = infer(img)
    boxes, scores, classes, nums = outputs["yolo_nms"], outputs["yolo_nms_1"], outputs["yolo_nms_2"], outputs["yolo_nms_3"]
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           scores[0][i].numpy(),
                                           boxes[0][i].numpy()))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
