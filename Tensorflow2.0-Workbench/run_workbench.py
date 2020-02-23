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

test_checkpoint = files.get_last_checkpoint()


def run_single_script():
    # just create class file
    if defaults.FLAGS.create_class_file:
        print("Gathering classifier data...")
        classifiers = files.get_classifiers(defaults.IMAGES_PATH)
        files.create_classifier_file(classifiers)
        print("\tData successfully classified!\n")

    # just sort images
    if defaults.FLAGS.sort_images:
        print("Sorting images...")
        files.sort_images(defaults.DEFAULT_NUM_VAL_IMAGES)
        print("\n\tAll images sorted!\n")

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

        # just detect images
        if defaults.FLAGS.detect_img:
            # generating tensorflow models
            print("\nTesting Images...")
            chkpnt_weights = files.get_last_checkpoint()
            if path.isfile(preferences.validate_input):
                print("\tTesting on image: " + preferences.validate_input + "\n")
                detect_img.run_detect(preferences.classifier_file,
                                      chkpnt_weights,
                                      preferences.tiny,
                                      preferences.image_size,
                                      preferences.validate_input,
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
    print("\nWelcome to the Digital Roll Workbench")
    print("\nEnter 'help' or 'h' for a list of commands:")

    while True:
        userInput = input("\n<WORKBENCH>: ")
        userInput.lower()
        userInput.strip()
        if userInput == "help" or userInput == "h":
            print("\n COMMANDS")
            print("\n help or h                      ==> Brings up this help display")
            print("\n run or r                       ==> Starts the process of training and validation")
            print("\n                                  + Saves the model at given output location")
            print("\n                                    and creates a Apple CoreML converted version")
            print("\n test or t <path to image>      ==> Tests a given image using the last checkpoint")
            print("\n display or d                   ==> Displays current settings")
            print("\n load or l <path to pref.txt>   ==> Loads a given .txt file as the current preference text")
            print("\n save or s <new .txt path>      ==> saves the current settings to the path + name given")
            print("\n                                  example: save C:\\Users\\new_pref.txt")
            print("\n change or c <variable> <value> ==> Changes the setting variable to a new value")
            print("\n                                  example: change batch_size 3")
            print("\n quit or q                      ==> Exits the Workbench")

        elif userInput == "run" or userInput == "r":
            run()

        elif userInput[0:5] == "test " or userInput[0:2] == "t ":
            error = False
            if userInput[0:2] == "t ":
                img_path = userInput[2:]
            else:
                img_path = userInput[5:]
            img_path.strip("\n\r")
            print("Searching for: " + img_path)
            if path.exists(os.getcwd().replace("\\", "/") + "/" + img_path):
                img_path = os.getcwd().replace("\\", "/") + "/" + img_path
                print("Found file at: " + img_path)
            if path.exists(img_path):
                # test to see if it is a single image or multiple
                print("\nTesting Images...")
                chkpnt_weights = files.get_last_checkpoint()
                if path.isfile(img_path):
                    print("\tTesting on image: " + img_path + "\n")
                    detect_img.run_detect(preferences.classifier_file,
                                          chkpnt_weights,
                                          preferences.tiny,
                                          preferences.image_size,
                                          img_path,
                                          preferences.output + img_path.split('/')[-1].split('.')[0] + "_output.jpg",
                                          preferences.num_classes)
                else:
                    for file in os.listdir(img_path):
                        if '.jpg' in file:
                            detect_img.run_detect(preferences.classifier_file,
                                                  chkpnt_weights,
                                                  preferences.tiny,
                                                  preferences.image_size,
                                                  img_path + file,
                                                  preferences.output + file.split('.')[0] + "_output.jpg",
                                                  preferences.num_classes)
                            print("\tTesting on image: " + img_path + file + "\n")
                print("\n\tImages Tested and stored in " + preferences.output)
            else:
                print("ERROR: Could not find " + img_path)


        elif userInput == "display" or userInput == "d":
            # Display pref
            print("\nCurrent Preferences:")
            preferences.print_pref()

        elif userInput[0:5] == "load " or userInput[0:2] == "l ":
            error = False
            if userInput[0:2] == "l ":
                pref_path = userInput[2:]
            else:
                pref_path = userInput[5:]
            pref_path.strip("\n\r")
            print("Searching for: " + pref_path)
            if path.exists(os.getcwd().replace("\\", "/") + "/" + pref_path):
                pref_path = os.getcwd().replace("\\", "/") + "/" + pref_path
                print("Found file at: " + pref_path)
            try:
                load(pref_path)
            except:
                print("ERROR: Loading failed, see above errors for more info")

        elif userInput[0:5] == "save " or userInput[0:2] == "s ":
            if userInput[0:2] == "s ":
                save_path = userInput[2:]
            else:
                save_path = userInput[5:]
            save_path.strip("\n\r")
            print("Attempting to save to " + save_path)
            if not path.exists(save_path):
                # open a new txt and copy in settings
                try:
                    save(save_path)
                    print("Successfully saved!")
                except:
                    print("ERROR: Failed to save")
            else:
                print("ERROR: File with this name already exists at this location")

        elif userInput[0:6] == "change " or userInput[0:2] == "c ":
            error = False
            userInputArr = userInput.split(" ")
            if len(userInputArr) == 3:
                try:
                    if userInputArr[1] == "batch_size":
                        try:
                            preferences.batch_size = int(userInputArr[2])
                        except:
                            print("ERROR: Please give an integer value")
                            error = True

                    elif userInputArr[1] == "checkpoints_path":
                        if path.exists(userInputArr[2]):
                            preferences.checkpoint_output = userInputArr[2]
                        else:
                            print("ERROR: Bad checkpoint output directory given")
                            error = True

                    elif userInputArr[1] == "test_checkpoint":
                        if path.exists(userInputArr[2]):
                            this.test_checkpoint = userInputArr[2]
                        else:
                            print("ERROR: Bad testing checkpoint directory given")
                            error = True

                    elif userInputArr[1] == "classifier_file":
                        old_classifier = preferences.classifier_file
                        preferences.classifier_file = userInputArr[2]
                        try:
                            preferences.num_classes = files.get_num_classes(os.getcwd().replace("\\", "/")
                                                                            + "/"
                                                                            + preferences.classifier_file[1:])
                        except:
                            print("ERROR: Failed to update classifier file, new file not found")
                            preferences.classifier_file = old_classifier
                            error = True

                    elif userInputArr[1] == "dataset_test":
                        if path.exists(userInputArr[2]):
                             preferences.dataset_test = userInputArr[2]
                        else:
                            print("ERROR: Bad testing checkpoint directory given")
                            error = True

                    elif userInputArr[1] == "epochs":
                        try:
                            preferences.epochs = int(userInputArr[2])
                        except:
                            print("ERROR: Please give an integer value")
                            error = True

                    elif userInputArr[1] == "image_size":
                        try:
                            preferences.image_size = int(userInputArr[2])
                        except:
                            print("ERROR: Please give an integer value")
                            error = True


                    elif userInputArr[1] == "learning_rate":
                        try:
                            preferences.learning_rate = float(userInputArr[2])
                        except:
                            print("ERROR: Please give an float value")
                            error = True

                    elif userInputArr[1] == "mode":
                        if userInputArr[2] == "fit" \
                                or userInputArr[2] == "eager_fit" \
                                or userInputArr[2] == "eager_tf":
                            preferences.mode = userInputArr[2]
                        else:
                            print(
                                "\nERROR: Bad mode value given, please choose one of the following")
                            print("\n       ==> fit, eager_fit, eager_tf")
                            error = True

                    elif userInputArr[1] == "output":
                        if path.exists(userInputArr[2]):
                             preferences.output = userInputArr[2]
                        else:
                            print("ERROR: Bad output directory given")
                            error = True

                    elif userInputArr[1] == "tiny":
                        try:
                            preferences.tiny = bool(userInputArr[2])
                        except:
                            print("ERROR: Please give an true or false")
                            error = True

                    elif userInputArr[1] == "transfer":
                        if userInputArr[2] == "none" \
                                or userInputArr[2] == "darknet" \
                                or userInputArr[2] == "no_output" \
                                or userInputArr[2] == "frozen" \
                                or userInputArr[2] == "fine_tune":
                            preferences.transfer = userInputArr[2]
                        else:
                            print(
                                "\nERROR: Bad transfer value given, please choose one of the following")
                            print("\n       ==> none, darknet, no_output, frozen, fine_tune")
                            error = True

                    elif userInputArr[1] == "validate_input":
                        if path.exists(userInputArr[2]):
                            preferences.validate_input = userInputArr[2]
                        else:
                            print("ERROR: Failed to find directory for validation")
                            error = True

                    elif userInputArr[1] == "weight_path":
                        old_weights = preferences.weights
                        preferences.weights = userInputArr[2]
                        try:
                            preferences.weight_num_classes = files.get_num_classes(os.getcwd().replace("\\", "/")
                                                                                   + "/"
                                                                                   + preferences.weights[1:])
                        except:
                            print("ERROR: Failed to update weights file, new file not found")
                            preferences.weights = old_weights
                            error = True

                    else:
                        print("ERROR: Unknown variable name")
                        error = True
                except:
                    print("ERROR: Failed to change variable to given value, try 'change ?' for a list of names")
                    error = True
                if not error:
                    print("Set " + userInputArr[1] + " to " + userInputArr[2])
            elif len(userInputArr) == 2 and userInputArr[1] == '?':
                print("\tBatch Size           ==> batch_size       ACCEPTS: int")
                print("\tCheckpoint Output    ==> checkpoints_path ACCEPTS: path to save checkpoints at")
                print("\tClassifier file      ==> classifier_file  ACCEPTS: path to file")
                print("\tNumber of Classes    ==X Automatically updated when classifier is changed")
                print("\tDataset test         ==> dataset_test     ACCEPTS: path to folder with images")
                print("\tDataset train        ==> dataset_train    ACCEPTS: path to folder")
                print("\tEpochs               ==> epochs           ACCEPTS: int")
                print("\tImage Size           ==> image_size       ACCEPTS: int")
                print("\tLearning Rate        ==> learning_rate    ACCEPTS: float")
                print("\tMode                 ==> mode             ACCEPTS: fit, eager_fit, eager_tf")
                print("\tOutput Model         ==> output           ACCEPTS: path to save location")
                print("\tTiny Weights         ==> tiny             ACCEPTS: true/false")
                print("\tTransfer             ==> transfer         ACCEPTS: none, darknet, no_output, frozen, fine_tune")
                print("\tValidate Image Input ==> validate_input   ACCEPTS: path to file")
                print("\tWeights Path         ==> weight_path      ACCEPTS: path to file")
                print("\tWeighted Classes     ==X Automatically updated when weights is changed")
                print("\tPreference File      ==X Automatically your most recently loaded preference file")
            else:
                print("Not enough arguments, please provide a variable and a value ie batch_size 3")

        elif userInput == "quit" or userInput == "q":
            break
        else:
            # end of cases, inform the user that their input was invalid
            print("\nCommand not recognized, try 'help' or 'h' for a list of options")


def load(pref_path):
    if path.exists(pref_path):
        # Set new preferences
        preferences.pref_file = pref_path
        with open(pref_path, "r") as f:
            for line in f.readlines():
                if defaults.BATCH_SIZE_VAR + "=" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    if path.exists(txt_input):
                        preferences.batch_size = int(txt_input)
                    else:
                        print("ERROR: Bad batch size given, cannot convert value to int")
                        error = True

                elif defaults.CHECKPOINT_VAR + "=" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    if path.exists(txt_input):
                        preferences.checkpoint_output = txt_input
                    else:
                        print("ERROR: Bad checkpoint save directory given")
                        error = True

                elif defaults.TEST_CHECKPOINT_VAR + "=" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    try:
                        this.test_checkpoint = txt_input
                    except:
                        print("ERROR: Bad testing checkpoint directory given")
                        error = True

                elif defaults.CLASSIFIERS_VAR + "=" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    old_classifier = preferences.classifier_file
                    preferences.classifier_file = txt_input
                    try:
                        preferences.num_classes = files.get_num_classes(preferences.classifier_file)
                    except:
                        try:
                            preferences.num_classes = files.get_num_classes(os.getcwd().replace("\\", "/")
                                                                            + "/"
                                                                            + preferences.classifier_file[1:])
                        except:
                            print("ERROR: Failed to update classifier file, new file not found")
                            preferences.classifier_file = old_classifier
                            error = True

                elif defaults.DATASET_TEST_VAR + "=" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    if path.exists(txt_input):
                        preferences.dataset_test = txt_input
                    else:
                        print("ERROR: Bad test dataset directory given")
                        error = True

                elif defaults.DATASET_TRAIN_VAR + "=" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    if path.exists(txt_input):
                        preferences.dataset_train = txt_input
                    else:
                        print("ERROR: Bad train dataset directory given")
                        error = True

                elif defaults.EPOCH_NUM_VAR + "=" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    try:
                        preferences.epochs = int(txt_input)
                    except:
                        print("ERROR: Bad epochs value given, cannot convert to int")
                        error = True

                elif defaults.IMAGE_SIZE_VAR + ":" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    try:
                        preferences.image_size = int(txt_input)
                    except:
                        print("ERROR: Bad image size value given, cannot convert to int")
                        error = True

                elif defaults.LEARN_RATE_VAR + ":" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    try:
                        preferences.image_size = float(txt_input)
                    except:
                        print("ERROR: Bad learning rate value given, cannot convert to float")
                        error = True

                elif defaults.MODE_VAR + "=" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    if txt_input == "fit" \
                            or txt_input == "eager_fit" \
                            or txt_input == "eager_tf":
                        preferences.mode = txt_input
                    else:
                        print("\nERROR: Bad mode value given, please update the file and choose one of the following")
                        print("\n       ==> fit, eager_fit, eager_tf")
                        error = True

                elif defaults.OUTPUT_VAR + "=" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    if path.exists(txt_input):
                        preferences.output = txt_input
                    else:
                        print("ERROR: Bad output directory given")
                        error = True

                elif defaults.TINY_WEIGHTS_VAR + ":" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    try:
                        preferences.tiny = bool(txt_input)
                    except:
                        print("ERROR: Failed to give True/False to tiny value")
                        error = True

                elif defaults.TRANSFER_VAR + "=" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    if txt_input == "none" \
                            or txt_input == "darknet" \
                            or txt_input == "no_output"\
                            or txt_input == "frozen"\
                            or txt_input == "fine_tune":
                        preferences.transfer = txt_input
                    else:
                        print("\nERROR: Bad transfer value given, please update the file and choose one of the following")
                        print("\n       ==> none, darknet, no_output, frozen, fine_tune")
                        error = True

                elif defaults.VALID_IN_VAR + "=" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    if path.exists(txt_input):
                        preferences.validate_input = txt_input
                    else:
                        print("ERROR: Failed to find directory for validation")
                        error = True

                elif defaults.WEIGHTS_PATH_VAR + "=" in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    if path.exists(txt_input):
                        old_weights = preferences.weights
                        preferences.weights = txt_input
                        try:
                            preferences.weight_num_classes = files.get_num_classes(
                                os.getcwd().replace("\\", "/")
                                + "/"
                                + preferences.weights[1:])
                        except:
                            print("ERROR: Failed to update weights file, new file not found")
                            preferences.weights = old_weights
                            error = True
                    else:
                        print("ERROR: Failed to update weights path")
                        error = True
        if error:
            print("A setting has failed to load properly please check above errors for info")
        else:
            print("\nNew Preferences:")
            preferences.print_pref()
    else:
        print("\nERROR: Bad Preferences File, could not find file")

def save(save_path):
    with open(save_path, "w") as f:
        f.write(defaults.BATCH_SIZE_VAR + "= " + str(preferences.batch_size) + "\n")
        f.write(defaults.CHECKPOINT_VAR + "= " + str(preferences.checkpoint_output) + "\n")
        f.write(defaults.TEST_CHECKPOINT_VAR + "= " + str(test_checkpoint) + "\n")
        f.write(defaults.CLASSIFIERS_VAR + "= " + str(preferences.classifier_file) + "\n")
        f.write(defaults.DATASET_TEST_VAR + "= " + str(preferences.dataset_test) + "\n")
        f.write(defaults.DATASET_TRAIN_VAR + "= " + str(preferences.dataset_train) + "\n")
        f.write(defaults.EPOCH_NUM_VAR + "= " + str(preferences.epochs) + "\n")
        f.write(defaults.IMAGE_SIZE_VAR + "= " + str(preferences.image_size) + "\n")
        f.write(defaults.LEARN_RATE_VAR + "= " + str(preferences.learning_rate) + "\n")
        f.write(defaults.MODE_VAR + "= " + str(preferences.mode) + "\n")
        f.write(defaults.OUTPUT_VAR + "= " + str(preferences.output) + "\n")
        f.write(defaults.TINY_WEIGHTS_VAR + "= " + str(preferences.tiny) + "\n")
        f.write(defaults.TRANSFER_VAR + "= " + str(preferences.transfer) + "\n")
        f.write(defaults.VALID_IN_VAR + "= " + str(preferences.validate_input) + "\n")
        f.write(defaults.WEIGHTS_CLASS_VAR + "= " + str(preferences.weight_num_classes) + "\n")


def run():

    # check if necessary files exist
    error = files.checkIfNecessaryPathsAndFilesExist()

    if not error:
        return
    # Run specified file
    run_single_script()

    # create classifiers.names
    print("\nGathering classifier data...")
    classifiers = files.get_classifiers(defaults.IMAGES_PATH)
    files.create_classifier_file(classifiers)
    print("\n\tData successfuly classified!\n")

    # sort all the images
    print("Sorting images...")
    files.sort_images(defaults.DEFAULT_NUM_VAL_IMAGES)
    print("\n\tAll images sorted!\n")

    # generate tf records
    print("Generating images and xml files into tfrecords...")
    generate_tf.generate_tfrecords(defaults.TRAIN_IMAGE_PATH,
                                  preferences.dataset_train)
    generate_tf.generate_tfrecords(defaults.TEST_IMAGE_PATH,
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
