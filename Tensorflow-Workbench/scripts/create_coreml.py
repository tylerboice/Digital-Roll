import coremltools
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import ResNet50
import tfcoreml
import tensorflow as tf
import coremltools
import coremltools.proto.FeatureTypes_pb2 as ft
import numpy as np
from yolov3_tf2.dataset import transform_images


from scripts import preferences


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
                           input_tensor=tf.keras.Input(shape=(224, 224, 3), name='image'))

    print("\n\tLoading up a .h5 from the weights " + weights + " for conversion...\n")
    keras_model.load_weights(weights).expect_partial()

    print("\n\tWeights Loaded!\n")

    keras_model.save(output + keras_model_name + ".h5")
    print("\n\tSaved a copy of the .h5 at " + output + keras_model_name + ".h5\n")

    # print input shape
    print("\t Using input shape: " + str(keras_model.input_shape))

    print("\n\tConverting CoreML Model from path: " + output + keras_model_name + ".h5\n")

    # get input, output node names for the TF graph from the Keras model
    input_name = keras_model.inputs[0].name.split(':')[0]
    keras_output_node_name = keras_model.outputs[0].name.split(':')[0]
    graph_output_node_name = keras_output_node_name.split('/')[-1]

    model = tfcoreml.convert(output + keras_model_name + ".h5",
                             image_input_names=input_name,
                             input_name_shape_dict={input_name: (1, 224, 224, 3)},
                             output_feature_names=[graph_output_node_name],
                             minimum_ios_deployment_target='13')
    # Set Metadata of madel to add clarity of use
    print("\n\tAdding metadata\n")
    model.author = 'Jason Robinson & Team Digital Roll'
    model.license = 'See https://github.com/tylerboice/Digital-Roll for more info.'
    model.short_description = 'A converted YOLO model for predicting polyhedral dice.'
    model.input_description['image'] = 'Input Image scene to be classified'
    model.output_description['Identity'] = 'Array of data contianing bounding boxes,' \
                                           ' scores, class label, and a prediction ID number'

    model.save(output + coreml_model_name + '.mlmodel')
    print("\n\tCoreML saved at " + output + coreml_model_name + ".mlmodel\n")
    spec = coremltools.utils.load_spec(output + coreml_model_name + '.mlmodel')
    print(spec.description)




    # This code apparently will only work if you have a Mac
    try:
        img = tf.image.decode_image(open('./data/d4-1 (18).jpg', 'rb').read(), channels=3)
        img = tf.expand_dims(img, 0)
        img = transform_images(img, 224)
        out_dict = model.predict({'image': img})
        print(str(out_dict))
    except Exception as e:
        print("Cannot test coreml models without a Mac device: " + str(e))



    EXTRA = False
    if EXTRA is True:
        print("\n\tStarting extra conversion step\n")
        spec = coremltools.utils.load_spec(output + coreml_model_name + '.mlmodel')
        print(spec.description)

        coreml_model = coremltools.models.MLModel(output + coreml_model_name + '.mlmodel')
        spec = coreml_model.get_spec()
        spec_layers = getattr(spec, spec.WhichOneof("Type")).layers

        print("\n\tRefactoring input...\n")
        input = spec.description.input[0]
        input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
        input.type.imageType.height = 224
        input.type.imageType.width = 224

        coremltools.utils.save_spec(spec, output + coreml_model_name + '.mlmodel')
        print(spec.description)

        print("\n\tRefactoring output...\n")
        # find the current output layer and save it for later reference
        last_layer = spec_layers[-1]

        # add the post-processing layer
        new_layer = spec_layers.add()
        new_layer.name = 'output'

        # Configure it as an activation layer
        new_layer.activation.linear.alpha = 255
        new_layer.activation.linear.beta = 0

        print("\n\tAdding in layering...\n")
        # Use the original model's output as input to this layer
        new_layer.input.append(last_layer.output[0])

        # Name the output for later reference when saving the model
        new_layer.output.append('classLabel')

        # Find the original model's output description
        output_description = next(x for x in spec.description.output if x.name == last_layer.output[0])

        # Update it to use the new layer as output
        output_description.name = new_layer.name
        print("\n\tMarking output...\n")
        # Mark the new layer as image
        layer_output = spec.description.output[0]
        layer_output.type.imageType.colorSpace = ft.ImageFeatureType.RGB
        layer_output.type.imageType.height = 224
        layer_output.type.imageType.width = 224
        print(spec.description)
        updated_model = coremltools.models.MLModel(spec)
        model_file_name = 'updated_model.mlmodel'
        print("\n\tSaving...\n")
        updated_model.save(output + model_file_name)


# Function to mark the layer as output
# https://forums.developer.apple.com/thread/81571#241998
def convert_multiarray_output_to_image(spec, feature_name, is_bgr=False):
    """
    Convert an output multiarray to be represented as an image
    This will modify the Model_pb spec passed in.
    Example:
        model = coremltools.models.MLModel('MyNeuralNetwork.mlmodel')
        spec = model.get_spec()
        convert_multiarray_output_to_image(spec,'imageOutput',is_bgr=False)
        newModel = coremltools.models.MLModel(spec)
        newModel.save('MyNeuralNetworkWithImageOutput.mlmodel')
    Parameters
    ----------
    spec: Model_pb
        The specification containing the output feature to convert
    feature_name: str
        The name of the multiarray output feature you want to convert
    is_bgr: boolean
        If multiarray has 3 channels, set to True for RGB pixel order or false for BGR
    """
    for output in spec.description.output:
        if output.name != feature_name:
            continue
        if output.type.WhichOneof('Type') != 'multiArrayType':
            raise ValueError("%s is not a multiarray type" % output.name)
        print("\nGetting Shape...")
        array_shape = tuple(output.type.multiArrayType.shape)
        channels, height, width = array_shape
        print("\nShape Acquired, adding color values...")
        from coremltools.proto import FeatureTypes_pb2 as ft
        if channels == 1:
            output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
        elif channels == 3:
            if is_bgr:
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('BGR')
            else:
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('RGB')
        else:
            raise ValueError("Channel Value %d not supported for image inputs" % channels)
        output.type.imageType.width = width
        output.type.imageType.height = height
