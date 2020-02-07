from scripts import defaults
from scripts import files
from scripts import generate_tf
from scripts import new_convert
from scripts import new_train


############################## MAIN ##########################
def main():

    #check if necessary files exist
    print("\nChecking that necessary file path exist...")
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
    generate_tf.generate_tfrecods(defaults.TRAIN_IMAGE_PATH, defaults.TRAIN_TF_RECORD_PATH)
    generate_tf.generate_tfrecods(defaults.TEST_IMAGE_PATH, defaults.TEST_TF_RECORD_PATH)
    print("\n\tSuccessfully generated tf records\n")

    print("Converting records to checkpoint...\n")
    new_convert.main()
    print("\nCheckpoint Converted!")

    print("\nBegin Training... \n")
    new_train.main()
    print("\n\tTraining Complete")
main()
