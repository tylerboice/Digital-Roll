import coremltools
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import ResNet50
import tfcoreml
import tensorflow as tf


from scripts import preferences
#from scripts import preferences

########################## EXPORT_COREML #############################
# Description: converts tensorflow model to coreml model and outputs it to output
# Parameters: ouptut - String - directory the tf model is in
# Return: Nothing
def export_coreml(output, weights):
    # input_shape_dict =
    # keras_model = MobileNet(weights=None, input_shape=(224, 224, 3))
    # keras_model.save(output, save_format='tf')
    # tf.saved_model.save(keras_model, './savedmodel')
    keras_model_name = "model"
    coreml_model_name = "core_ml"

    keras_model = ResNet50(weights=None,
                           input_tensor=tf.keras.Input(shape=(224, 224, 3), name='Image'),
                           input_shape=(224, 224, 3))

    print("\n\tLoading up a .h5 from the weights " + weights + " for conversion...\n")
    keras_model.load_weights(weights).expect_partial()

    print("\n\tWeights Loaded!\n")

    keras_model.save(output + keras_model_name + ".h5")
    print("\n\tSaved a copy of the .h5 at " + output + keras_model_name + ".h5\n")

    # print input shape
    print("\t Using input shape: " + str(keras_model.input_shape))

    print("\n\tConverting CoreML Model from path: " + output + coreml_model_name + ".h5\n")

    # get input, output node names for the TF graph from the Keras model
    input_name = keras_model.inputs[0].name.split(':')[0]
    keras_output_node_name = keras_model.outputs[0].name.split(':')[0]
    graph_output_node_name = keras_output_node_name.split('/')[-1]

    model = tfcoreml.convert(output + keras_model_name + ".h5",
                             input_name_shape_dict={input_name: (1, 224, 224, 3)},
                             output_feature_names=[graph_output_node_name],
                             minimum_ios_deployment_target='13')
    # Set Metadata of madel to add clarity of use
    model.author = 'Jason Robinson & Team Digital Roll'
    model.license = 'See https://github.com/tylerboice/Digital-Roll for more info.'
    model.short_description = 'A converted YOLO model for predicting polyhedral dice.'
    model.input_description['Image'] = 'Input Image scene to be classified'
    model.output_description['Identity'] = 'Array of data contianing bounding boxes,' \
                                           ' scores, class label, and a prediction ID number'

    model.save(output + coreml_model_name + '.mlmodel')
    print("\n\tCoreML saved at " + output + coreml_model_name + ".mlmodel\n")
