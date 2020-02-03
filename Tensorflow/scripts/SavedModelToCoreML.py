from tensorflow.keras.applications import MobileNet
import tfcoreml

keras_model = MobileNet(weights=None, input_shape=(224, 224, 3))
keras_model.save('./savedmodel', save_format='tf')
# tf.saved_model.save(keras_model, './savedmodel')

model = tfcoreml.convert('./savedmodel',
                         mlmodel_path='./model.mlmodel',
                         input_name_shape_dict={'input_1': (1, 224, 224, 3)},
                         output_feature_names=['Identity'],
                         minimum_ios_deployment_target='13')