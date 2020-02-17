import os
import sys
from os import path

from scripts import defaults
from scripts import files
from scripts import preferences
from scripts import generate_tf
from scripts import convert_weights
from scripts import train_workbench
from scripts import create_tf_model
from scripts import detect_img
from scripts import create_coreml



def run_single_script():
    # just create class file
    if defaults.FLAGS.create_class_file:
        print("Gathering classifier data...")
        classifiers = files.get_classifiers(defaults.IMAGES_PATH)
        files.create_classifier_file(classifiers)
        print("\tData successfuly classified!\n")
        exit()

    # just sort images
    if defaults.FLAGS.sort_images:
        print("Sorting images...")
        files.sort_images(defaults.DEFAULT_NUM_VAL_IMAGES)
        print("\n\tAll images sorted!\n")
        exit()

    # just generate tf records
    if defaults.FLAGS.generate_tf:
        classifiers = files.get_classifiers(defaults.IMAGES_PATH)
        files.create_classifier_file(classifiers)
        print("Generating images and xml files into tfrecords...")
        generate_tf.generate_tfrecods(defaults.TRAIN_IMAGE_PATH,
                                      preferences.dataset_train)
        generate_tf.generate_tfrecods(defaults.TEST_IMAGE_PATH,
                                      preferences.dataset_test)
        print("\n\tSuccessfully generated tf records\n")
        exit()

    # just convert weights
    if defaults.FLAGS.convert_weight:
        if not os.path.exists(preferences.weights):
            print("Weights file does not exist")
            exit()
        print("Converting records to checkpoint...\n")
        convert_weights.run_weight_convert(preferences.weights,
                                           preferences.checkpoint_output,
                                           preferences.tiny,
                                           preferences.weight_num_classes)
        exit()

    # just train
    if defaults.FLAGS.train:
        print("\nBegin Training... \n")
        train_workbench.run_train(preferences.dataset_train,
                                  preferences.dataset_test,
                                  preferences.tiny,
                                  preferences.weights,
                                  preferences.classifier_file,
                                  preferences.mode,
                                  preferences.transfer,
                                  preferences.image_size,
                                  preferences.epochs,
                                  preferences.batch_size,
                                  preferences.learning_rate,
                                  preferences.num_classes,
                                  preferences.weight_num_classes)
        print("\n\tTraining Complete!")
        exit()

    # just create tf model
    if defaults.FLAGS.tf_model:
        # generating tensorflow models
        print("\nGenerating TensorFlow model...")
        chkpnt_weights = files.get_last_checkpoint()
        print("\n\tUsing checkpoint: " + chkpnt_weights + "\n")
        if path.isfile(preferences.validate_input):
            create_tf_model.run_export_tfserving(chkpnt_weights,
                                                      preferences.tiny,
                                                      preferences.output,
                                                      preferences.classifier_file,
                                                      preferences.validate_input + file,
                                                      preferences.num_classes)
        else:
            model_saved = False
            for file in os.listdir(preferences.validate_input):
                if '.jpg' in file and not model_saved:
                    create_tf_model.run_export_tfserving(chkpnt_weights,
                                                              preferences.tiny,
                                                              preferences.output,
                                                              preferences.classifier_file,
                                                              preferences.validate_input + file,
                                                              preferences.num_classes)
                    model_saved = True
        print("\n\tTensorFlow model Generated!")
        exit()

    # just detect images
    if defaults.FLAGS.detect_img:
        # generating tensorflow models
        print("\nTesting Images...")
        chkpnt_weights = files.get_last_checkpoint()
        if path.isfile(preferences.validate_input):
            print("\tTesting on image: " + file + "\n")
            detect_img.run_detect(preferences.classifier_file,
                                   chkpnt_weights,
                                   preferences.tiny,
                                   preferences.image_size,
                                   preferences.validate_input + file,
                                   preferences.output,
                                   preferences.num_classes)
        else:
            for file in os.listdir(preferences.validate_input):
                if '.jpg' in file:
                    detect_img.run_detect(preferences.classifier_file,
                                           chkpnt_weights,
                                           preferences.tiny,
                                           preferences.image_size,
                                           preferences.validate_input + file,
                                           preferences.output + file + "_output.jpg",
                                           preferences.num_classes)
                    print("\tTesting on image: " + preferences.validate_input + file + "\n")
        print("\n\tImages Tested and stored in " + preferences.output)

    # just export coreml model
    if defaults.FLAGS.core_ml:
        print("Create a CoreML model...")
        create_coreml.export_coreml(preferences.output)
        print("Core ML model created!")


############################## MAIN ##########################
def main():
    print("\nWelcome to the workbench")
    # get help if needed
    defaults.get_help()
    # Display pref
    print("\nCurrent Preferences:")
    preferences.print_pref()
    while True:
        userInput = input("\n<WORKBENCH>: ")
        userInput.lower()
        userInput.strip()
        if userInput == "help" or userInput == "h":
            print("\n COMMANDS")
            print("\n help or h == Brings up this help display")
            print("\n run or r == Starts the process of training and validation")
            print("\n settings or s == Displays current settings")
            print("\n load or l <path to pref.txt> == Loads a given .txt file as the current preference text")
            print("\n change or c <variable> <value> == Changes the setting variable to a new value")
            print("\n quit or q == Exits the Workbench")

        elif userInput == "run" or userInput == "r":
            run()

        elif userInput == "settings" or userInput == "s":
            # Display pref
            print("\nCurrent Preferences:")
            preferences.print_pref()

        elif userInput[0:5] == "load " or userInput[0:2] == "l ":
            pref_path = userInput[5:]
            if path.exists(os.getcwd() + "/" + pref_path):
                pref_path = os.getcwd() + "/" + pref_path
            if path.exists(pref_path):
                #TODO Set new preferences
                print("\nNew Preferences:")
                preferences.print_pref()
            else:
                print("\nERROR: Bad Preferences File")

        elif userInput[0:6] == "change " or userInput[0:2] == "c ":
            userInputArr = userInput.split(" ")
            if len(userInputArr) == 3:
                try:
                    if userInputArr[1] == "batch_size":
                        preferences.batch_size = userInputArr[2]
                    elif userInputArr[1] == "checkpoint_output":
                        preferences.checkpoint_output = userInputArr[2]
                    elif userInputArr[1] == "classifier_file":
                        preferences.classifier_file = userInputArr[2]
                    elif userInputArr[1] == "dataset_test":
                        preferences.dataset_test = userInputArr[2]
                    elif userInputArr[1] == "epochs":
                        preferences.epochs = userInputArr[2]
                    elif userInputArr[1] == "image_size":
                        preferences.image_size = userInputArr[2]
                    elif userInputArr[1] == "learning_rate":
                        preferences.learning_rate = userInputArr[2]
                    elif userInputArr[1] == "mode":
                        preferences.mode = userInputArr[2]
                    elif userInputArr[1] == "num_classes":
                        preferences.num_classes = userInputArr[2]
                    elif userInputArr[1] == "output":
                        preferences.output = userInputArr[2]
                    elif userInputArr[1] == "tiny":
                        preferences.tiny = userInputArr[2]
                    elif userInputArr[1] == "transfer":
                        preferences.transfer = userInputArr[2]
                    elif userInputArr[1] == "validate_input":
                        preferences.validate_input = userInputArr[2]
                    elif userInputArr[1] == "weight_num_classes":
                        preferences.weight_num_classes = userInputArr[2]
                    else:
                        print("ERROR: Unknown variable name")
                except:
                    print("ERROR: Failed to change variable to given value")

                print("Set " + userInputArr[1] + " to " + userInputArr[2])

            else:
                print("Not enough arguments, please provide a variable and a value ie batch_size 3")



        elif userInput == "quit" or userInput == "q":
            break
        else:
            # end of cases, inform the user that their input was invalid
            print("\nCommand not recognized, try 'help' or 'h' for a list of options")



def run():

    # check if necessary files exist
    print("\n\nChecking that necessary file path exist...")
    files.checkIfNecessaryPathsAndFilesExist()


    # Run specified file
    run_single_script()

    print("\tAll necessary files exist!\n")

    # create classifiers.names
    print("Gathering classifier data...")
    classifiers = files.get_classifiers(defaults.IMAGES_PATH)
    files.create_classifier_file(classifiers)
    print("\tData successfuly classified!\n")

    # sort all the images
    print("Sorting images...")
    files.sort_images(defaults.DEFAULT_NUM_VAL_IMAGES)
    print("\n\tAll images sorted!\n")

    # generate tf records
    print("Generating images and xml files into tfrecords...")
    generate_tf.generate_tfrecods(defaults.TRAIN_IMAGE_PATH,
                                  preferences.dataset_train)
    generate_tf.generate_tfrecods(defaults.TEST_IMAGE_PATH,
                                  preferences.dataset_test)
    print("\n\tSuccessfully generated tf records\n")

    # convert to checkpoint
    print("Converting records to checkpoint...\n")
    convert_weights.run_weight_convert(preferences.weights,
                                       preferences.checkpoint_output,
                                       preferences.tiny,
                                       preferences.weight_num_classes)
    print("\nCheckpoint Converted!")

    # train
    print("\nBegin Training... \n")
    train_workbench.run_train(preferences.dataset_train,
                              preferences.dataset_test,
                              preferences.tiny,
                              preferences.weights,
                              preferences.classifier_file,
                              preferences.mode,
                              preferences.transfer,
                              preferences.image_size,
                              preferences.epochs,
                              preferences.batch_size,
                              preferences.learning_rate,
                              preferences.num_classes,
                              preferences.weight_num_classes)
    print("\n\tTraining Complete!")

    # generating tensorflow models
    print("\nGenerating TensorFlow model...")
    chkpnt_weights = files.get_last_checkpoint()
    print("\n\tUsing checkpoint: " + chkpnt_weights + "\n")
    if path.isfile(preferences.validate_input):
        create_tf_model.run_export_tfserving(chkpnt_weights,
                                                  preferences.tiny,
                                                  preferences.output,
                                                  preferences.classifier_file,
                                                  preferences.validate_input + file,
                                                  preferences.num_classes)
    else:
        model_saved = False
        for file in os.listdir(preferences.validate_input):
            if '.jpg' in file and not model_saved:
                create_tf_model.run_export_tfserving(chkpnt_weights,
                                                          preferences.tiny,
                                                          preferences.output,
                                                          preferences.classifier_file,
                                                          preferences.validate_input + file,
                                                          preferences.num_classes)
                model_saved = True
    print("\n\tTensorFlow model Generated!")

    # generating tensorflow models
    print("\nTesting Images...")
    chkpnt_weights = files.get_last_checkpoint()
    if path.isfile(preferences.validate_input):
        print("\tTesting on image: " + file + "\n")
        detect_img.run_detect(preferences.classifier_file,
                               chkpnt_weights,
                               preferences.tiny,
                               preferences.image_size,
                               preferences.validate_input + file,
                               preferences.output,
                               preferences.num_classes)
    else:
        for file in os.listdir(preferences.validate_input):
            if '.jpg' in file:
                detect_img.run_detect(preferences.classifier_file,
                                       chkpnt_weights,
                                       preferences.tiny,
                                       preferences.image_size,
                                       preferences.validate_input + file,
                                       preferences.output + file + "_output.jpg",
                                       preferences.num_classes)
                print("\tTesting on image: " + preferences.validate_input + file + "\n")
    print("\n\tImages Tested and stpreferences.ored in " + preferences.output)

    print("\nCreate a CoreML model...")
    create_coreml.export_coreml(preferences.output)
    print("\n\tCore ML model created!")

    print("\nWorkbench Successful!")
    print("\n\tAll models and images saved in " + preferences.output)


main()
