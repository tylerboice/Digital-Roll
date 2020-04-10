import coremltools
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import ResNet50
import tfcoreml
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

    keras_model = ResNet50(weights=None, input_shape=(224, 224, 3))

    print("Loading up a .h5 from the weights " + weights + " for conversion...")
    keras_model.load_weights(weights).expect_partial()

    print("Weights Loaded!")

    keras_model.save(output + "model.h5")
    print("Saved a copy of the .h5 at " + output + "model.h5")

    # print input shape
    print(keras_model.input_shape)

    print("Converting CoreML Model from path: " + output + "core_model.h5")

    # get input, output node names for the TF graph from the Keras model
    input_name = keras_model.inputs[0].name.split(':')[0]
    keras_output_node_name = keras_model.outputs[0].name.split(':')[0]
    graph_output_node_name = keras_output_node_name.split('/')[-1]

    model = tfcoreml.convert(output + "model.h5",
                             input_name_shape_dict={input_name: (1, 224, 224, 3)},
                             output_feature_names=[graph_output_node_name],
                             minimum_ios_deployment_target='13')

    model.save(output + 'core_model.mlmodel')
    print("CoreML saved at " + output + "core_model.h5")
