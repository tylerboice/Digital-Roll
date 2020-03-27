import coremltools
from tensorflow.keras.applications import MobileNet
import tfcoreml
#from scripts import preferences

########################## EXPORT_COREML #############################
# Description: converts tensorflow model to coreml model and outputs it to output
# Parameters: ouptut - String - directory the tf model is in
# Return: Nothing
def export_coreml(output):
    # input_shape_dict =
    # keras_model = MobileNet(weights=None, input_shape=(224, 224, 3))
    # keras_model.save(output, save_format='tf')
    # tf.saved_model.save(keras_model, './savedmodel')

    print("Converting CoreML Model from path: " + output)

    model = tfcoreml.convert(tf_model_path=output,
                             mlmodel_path=output + "coreML_model.mlmodel",
                             input_name_shape_dict={'input_28': (1, 224, 224, 3)},
                             output_feature_names=['Identity'],
                             minimum_ios_deployment_target='13')
