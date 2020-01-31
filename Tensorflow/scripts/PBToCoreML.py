import tfcoreml
import tensorflow as tf

frozen_model_file = 'frozen_inference_graph.pb'
input_tensor_shapes = {"input/placeholder:0": [1, 32, 32, 9]}
# Output CoreML model path
coreml_model_file = './model.mlmodel'
output_tensor_names = ['output/prediction:0']
def convert():
    # Read the pb model
    with tf.gfile.GFile(frozen_model_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # Then, we import the graph_def into a new Graph
    tf.import_graph_def(graph_def, name="")
    # Convert
    tfcoreml.convert(
        tf_model_path=frozen_model_file,
        mlmodel_path=coreml_model_file,
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names,
        image_scale=2/224.0,
        red_bias=-1,
        green_bias=-1,
        blue_bias=-1)
convert()


