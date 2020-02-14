import os
from os import path

from scripts import defaults
from scripts import files
from scripts import new_export_tfserving

batch_size = defaults.get_batch_size()
chkpnt_output = defaults.get_checkpoint_path()
classes = defaults.FLAGS.classifiers
dataset_test = defaults.get_dataset_test()
dataset_train = defaults.get_dataset_train()
epochs = defaults.get_epoch_num()
learning_rate = defaults.get_learn_rate()
mode = defaults.FLAGS.mode
num_classes = defaults.get_num_classes(classes)
output_model = defaults.get_model_output()
size = defaults.get_image_size()
tiny = defaults.is_tiny_weight()
transfer = defaults.get_transfer_type()
valid_image_input = defaults.get_valid_image_input()
weights_num_classes = defaults.FLAGS.weights_num_classes
weights = defaults.get_weights_path()

############################## MAIN ##########################
def main():

    # generating tensorflow models
    print("\nGenerating TensorFlow model...")
    chkpnt_weights = files.get_last_checkpoint()
    if path.isfile(valid_image_input):
        new_export_tfserving.run_export_tfserving = (chkpnt_weights,
                                                     tiny,
                                                     output_model,
                                                     classes,
                                                     valid_image_input,
                                                     num_classes)
    else:
        for file in os.listdir(valid_image_input):
            print( "---------------------------------------------------\nchkweight:" + chkpnt_weights + "\ntiny:" + str(tiny) + "\nout_model:" + output_model + "\nclasses" + str(classes)  + "\nfile" +  file  + "\nnum_classes" +  str(num_classes))
            if ".jpg" in file:
                new_export_tfserving.run_export_tfserving = (chkpnt_weights,
                                                             tiny,
                                                             output_model,
                                                             classes,
                                                             file,
                                                             num_classes)
    print("\n\tTensorFlow model Generated!")

main()
