import os
from os import path

from scripts import defaults
from scripts import files
from scripts import generate_tf
from scripts import new_convert
from scripts import new_train
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

    # Display pref
    print("\nCurrent Preferences:")
    defaults.print_pref()

    # check if necessary files exist
    print("\n\nChecking that necessary file path exist...")
    files.checkIfNecessaryPathsAndFilesExist()
    print("\tAll necessary files exist!\n")

    # create classifiers.names
    print("Gathering classifier data...")
    classifiers = files.get_classifiers(defaults.IMAGES_PATH)
    files.create_classifier_file(classifiers)
    print("\tData successfuly classified!\n")

    # sort all the images
    print("Sorting images...")
    files.sort_images()
    print("\n\tAll images sorted!\n")

    # generate tf records
    print("Generating images and xml files into tfrecords...")
    generate_tf.generate_tfrecods(defaults.TRAIN_IMAGE_PATH,
                                  defaults.TRAIN_TF_RECORD_PATH)
    generate_tf.generate_tfrecods(defaults.TEST_IMAGE_PATH,
                                  defaults.TEST_TF_RECORD_PATH)
    print("\n\tSuccessfully generated tf records\n")

    # convert to checkpoint
    print("Converting records to checkpoint...\n")
    new_convert.run_weight_convert(weights,
                                   chkpnt_output,
                                   tiny,
                                   weights_num_classes)
    print("\nCheckpoint Converted!")

    # train
    print("\nBegin Training... \n")
    '''
    new_train.run_train(dataset_train,
                        dataset_test,
                        tiny,
                        weights,
                        classifiers,
                        mode,
                        transfer,
                        size,
                        epochs,
                        batch_size,
                        learning_rate,
                        num_classes,
                        weights_num_classes)
    '''
    print("\n\tTraining Complete!")

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
