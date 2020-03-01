import os
import sys
from os import path

from scripts import defaults
from scripts import files
from scripts import print_to_terminal
from scripts import preferences
from scripts import generate_tf
from scripts import convert_weights
from scripts import train_workbench
from scripts import create_tf_model
from scripts import detect_img
from scripts import create_coreml

test_checkpoint = files.get_last_checkpoint(preferences.checkpoint_output)
SPLIT_CHAR = "="
START = 1001
CONTINUE = 1002
SINGLE = 1003

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def run_single_script():
    # just create class file
    if defaults.FLAGS.create_class_file:
        print("Gathering classifier data...")
        classifiers = files.get_classifiers(defaults.IMAGES_PATH)
        files.create_classifier_file(preferences.classifier_file, classifiers)
        print("\tData successfully classified!\n")

    # just sort images
    if defaults.FLAGS.sort_images:
        print("Sorting images...")
        files.sort_images(defaults.DEFAULT_NUM_VAL_IMAGES,
                          defaults.IMAGES_PATH,
                          defaults.TEST_IMAGE_PATH,
                          defaults.TRAIN_IMAGE_PATH,
                          defaults.VALIDATE_IMAGE_PATH
                          )
        print("\n\tAll images sorted!\n")

    # just generate tf records
    if defaults.FLAGS.generate_tf:
        classifiers = files.get_classifiers(defaults.IMAGES_PATH)
        files.create_classifier_file(preferences.classifier_file, classifiers)
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
        files.save_checkpoints(preferences.checkpoint_output, preferences.sessions, preferences.max_saved_sess)
        convert_weights.run_weight_convert(preferences.weights,
                                           preferences.checkpoint_output,
                                           preferences.tiny,
                                           preferences.weight_num_classes)
        print("Converted records to checkpoint!\n")
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
                                  preferences.weight_num_classes,
                                  preferences.checkpoint_output,
                                  preferences.max_checkpoints )

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


def load(pref_path):
    if path.exists(pref_path):

        # Set new preferences
        preferences.pref_file = pref_path
        with open(pref_path, "r") as f:
            for line in f.readlines():
                txt_input = files.get_input(line, SPLIT_CHAR)
                if defaults.BATCH_SIZE_VAR + SPLIT_CHAR in line:
                    if path.exists(txt_input):
                        preferences.batch_size = int(txt_input)
                    else:
                        print("ERROR: Bad batch size given, cannot convert value to int")
                        error = True

                elif defaults.CHECKPOINT_VAR + SPLIT_CHAR in line:
                    if path.exists(txt_input):
                        preferences.checkpoint_output = txt_input
                    else:
                        print("ERROR: Bad checkpoint save directory given")
                        error = True

                elif defaults.TEST_CHECKPOINT_VAR + SPLIT_CHAR in line:
                    try:
                        this.test_checkpoint = txt_input
                    except:
                        print("ERROR: Bad testing checkpoint directory given")
                        error = True

                elif defaults.CLASSIFIERS_VAR + SPLIT_CHAR in line:
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

                elif defaults.DATASET_TEST_VAR + SPLIT_CHAR in line:
                    if path.exists(txt_input):
                        preferences.dataset_test = txt_input
                    else:
                        print("ERROR: Bad test dataset directory given")
                        error = True

                elif defaults.DATASET_TRAIN_VAR + SPLIT_CHAR in line:
                    if path.exists(txt_input):
                        preferences.dataset_train = txt_input
                    else:
                        print("ERROR: Bad train dataset directory given")
                        error = True

                elif defaults.EPOCH_NUM_VAR + SPLIT_CHAR in line:
                    try:
                        preferences.epochs = int(txt_input)
                    except:
                        print("ERROR: Bad epochs value given, cannot convert to int")
                        error = True

                elif defaults.IMAGE_SIZE_VAR + SPLIT_CHAR in line:
                    try:
                        preferences.image_size = int(txt_input)
                    except:
                        print("ERROR: Bad image size value given, cannot convert to int")
                        error = True

                elif defaults.LEARN_RATE_VAR + SPLIT_CHAR in line:
                    try:
                        preferences.image_size = float(txt_input)
                    except:
                        print("ERROR: Bad learning rate value given, cannot convert to float")
                        error = True

                elif defaults.MAX_CHECK_VAR + SPLIT_CHAR in line:
                    try:
                        preferences.max_checkpoints = int(txt_input)
                    except:
                        print("ERROR: Bad max check value given, cannot convert to int")
                        error = True

                elif defaults.MAX_SAVED_SESS_VAR + SPLIT_CHAR in line:
                    try:
                        preferences.max_saved_sess = int(txt_input)
                    except:
                        print("ERROR: Bad max saved sessions value given, cannot convert to int")
                        error = True

                elif defaults.MODE_VAR + SPLIT_CHAR in line:
                    if txt_input == "fit" \
                            or txt_input == "eager_fit" \
                            or txt_input == "eager_tf":
                        preferences.mode = txt_input
                    else:
                        print("\nERROR: Bad mode value given, please update the file and choose one of the following")
                        print("\n       ==> fit, eager_fit, eager_tf")
                        error = True

                elif defaults.OUTPUT_VAR + SPLIT_CHAR in line:
                    if path.exists(txt_input):
                        preferences.output = txt_input
                    else:
                        print("ERROR: Bad output directory given")
                        error = True

                elif defaults.TINY_WEIGHTS_VAR + SPLIT_CHAR in line:
                    try:
                        preferences.tiny = bool(txt_input)
                    except:
                        print("ERROR: Failed to give True/False to tiny value")
                        error = True

                elif defaults.TRANSFER_VAR + SPLIT_CHAR in line:
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
            print_to_terminal.current_pref()
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
        f.write(defaults.MAX_CHECK_VAR + "= " + str(preferences.max_checkpoints) + "\n")
        f.write(defaults.MAX_SAVED_SESS_VAR + "= " + str(preferences.max_saved_sess) + "\n")
        f.write(defaults.OUTPUT_VAR + "= " + str(preferences.output) + "\n")
        f.write(defaults.TINY_WEIGHTS_VAR + "= " + str(preferences.tiny) + "\n")
        f.write(defaults.TRANSFER_VAR + "= " + str(preferences.transfer) + "\n")
        f.write(defaults.VALID_IN_VAR + "= " + str(preferences.validate_input) + "\n")
        f.write(defaults.WEIGHTS_CLASS_VAR + "= " + str(preferences.weight_num_classes) + "\n")


def run(start_from):

    single_script = False
    # check if necessary files exist
    error = files.checkIfNecessaryPathsAndFilesExist(defaults.IMAGES_PATH,
                                                     defaults.MIN_IMAGES,
                                                     defaults.OUTPUT_PATH,
                                                     defaults.TEST_IMAGE_PATH,
                                                     defaults.TRAIN_IMAGE_PATH,
                                                     defaults.VALIDATE_IMAGE_PATH,
                                                     defaults.YOLO_PATH)

    if not error:
        return

    # run was called, start from beginning
    if start_from == START:
        # create classifiers.names
        print("\nGathering classifier data...")
        classifiers = files.get_classifiers(defaults.IMAGES_PATH)
        files.create_classifier_file(preferences.classifier_file, classifiers)
        print("\n\tData successfuly classified!\n")

        # sort all the images
        print("Sorting images...")
        files.sort_images(defaults.DEFAULT_NUM_VAL_IMAGES,
                          defaults.IMAGES_PATH,
                          defaults.TEST_IMAGE_PATH,
                          defaults.TRAIN_IMAGE_PATH,
                          defaults.VALIDATE_IMAGE_PATH
                          )
        print("\n\tAll images sorted!\n")

        # generate tf records
        print("Generating images and xml files into tfrecords...\n")
        generate_tf.generate_tfrecords(defaults.TRAIN_IMAGE_PATH,
                                      preferences.dataset_train)
        generate_tf.generate_tfrecords(defaults.TEST_IMAGE_PATH,
                                      preferences.dataset_test)
        print("\n\tSuccessfully generated tf records!")

        # save previous sessions
        print("\nChecking for previous Sessions...\n")
        files.save_checkpoints(preferences.checkpoint_output, defaults.SAVED_SESS_PATH, preferences.max_saved_sess)
        print("\n\tDone!")

        # convert to checkpoint
        print("\nConverting records to checkpoint...\n")
        blockPrint()
        convert_weights.run_weight_convert(preferences.weights,
                                           preferences.checkpoint_output,
                                           preferences.tiny,
                                           preferences.weight_num_classes)
        enablePrint()

        print("\tCheckpoint Converted!")

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
                                  preferences.weight_num_classes,
                                  preferences.checkpoint_output,
                                  preferences.max_checkpoints )
        print("\n\tTraining Complete!")


    # generating tensorflow models
    print("\nGenerating TensorFlow model...")
    try:
        chkpnt_weights = files.get_last_checkpoint(preferences.checkpoint_output)
    except:
        chkpnt_weights = preferences.checkpoint_output

    if os.path.exists(chkpnt_weights):
        print("\nERROR: checkpoint_output " + chkpnt_weights + " does not exist\n")
        return

    print("\n\tUsing checkpoint: " + chkpnt_weights + "\n")

    if path.isfile(preferences.validate_input):
        create_tf_model.run_export_tfserving(chkpnt_weights,
                                                  preferences.tiny,
                                                  preferences.output,
                                                  preferences.classifier_file,
                                                  preferences.validate_input,
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
    if path.isfile(preferences.validate_input):
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
    print("\n\tImages Tested and stpreferences.ored in " + preferences.output)

    print("\nCreating a CoreML model...")
    blockPrint()
    create_coreml.export_coreml(preferences.output)
    enablePrint()
    print("\n\tCore ML model created!")

    print("\n=============================== Workbench Successful! ===============================")
    print("\n\tAll models and images saved in " + preferences.output)



############################## MAIN ##########################
def main():
    print("\nWelcome to the Digital Roll Workbench")
    print("\nEnter 'help' or 'h' for a list of commands:")
    running = True
    while running:
        try:
            userInput = input("\n<WORKBENCH>: ")
            userInput.lower()
            userInput.strip()
            if userInput == "help" or userInput == "h":
               print_to_terminal.help()

            elif userInput == "run" or userInput == "r":
                run(START)

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

            elif userInput == "continue" or userInput == "c":
                run(CONTINUE)

            elif userInput == "display" or userInput == "d":
                print_to_terminal.current_pref()

            elif userInput == "info" or userInput == "i":
                print_to_terminal.info()

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

            elif userInput[0:7] == "modify " or userInput[0:2] == "m ":
                error = False
                userInputArr = userInput.split(" ")
                if len(userInputArr) == 3:
                    try:
                        if userInputArr[1] == defaults.BATCH_SIZE_VAR:
                            try:
                                preferences.batch_size = int(userInputArr[2])
                            except:
                                print("ERROR: " + defaults.BATCH_SIZE_VAR + " taks an integer value")
                                error = True

                        elif userInputArr[1] == defaults.CHECKPOINT_VAR:
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

                        elif userInputArr[1] == defaults.CLASSIFIERS_VAR:
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

                        elif userInputArr[1] == defaults.DATASET_TEST_VAR:
                            if path.exists(userInputArr[2]):
                                 preferences.dataset_test = userInputArr[2]
                            else:
                                print("ERROR: Bad dataset test directory given")
                                error = True

                        elif userInputArr[1] == defaults.DATASET_TRAIN_VAR:
                            if path.exists(userInputArr[2]):
                                 preferences.dataset_train = userInputArr[2]
                            else:
                                print("ERROR: Bad dataset train directory given")
                                error = True

                        elif userInputArr[1] == defaults.EPOCH_NUM_VAR:
                            try:
                                preferences.epochs = int(userInputArr[2])
                            except:
                                print("ERROR: Please give an integer value")
                                error = True

                        elif userInputArr[1] == defaults.IMAGE_SIZE_VAR:
                            try:
                                preferences.image_size = int(userInputArr[2])
                            except:
                                print("ERROR: Please give an integer value")
                                error = True


                        elif userInputArr[1] == defaults.LEARN_RATE_VAR:
                            try:
                                preferences.learning_rate = float(userInputArr[2])
                            except:
                                print("ERROR: Please give an float value")
                                error = True

                        elif userInputArr[1] == defaults.MAX_CHECK_VAR:
                            try:
                                preferences.max_checkpoints = int(userInputArr[2])
                            except:
                                print("ERROR: Please give an integer value")
                                error = True

                        elif userInputArr[1] == defaults.MAX_SESS_VAR:
                            try:
                                preferences.max_sessions = int(userInputArr[2])
                            except:
                                print("ERROR: Please give an integer value")
                                error = True

                        elif userInputArr[1] == defaults.MODE_VAR:
                            if userInputArr[2] == "fit" \
                                    or userInputArr[2] == "eager_fit" \
                                    or userInputArr[2] == "eager_tf":
                                preferences.mode = userInputArr[2]
                            else:
                                print(
                                    "\nERROR: Bad mode value given, please choose one of the following")
                                print("\n       ==> fit, eager_fit, eager_tf")
                                error = True

                        elif userInputArr[1] == defaults.MODE_VAR:
                            if path.exists(userInputArr[2]):
                                 preferences.output = userInputArr[2]
                            else:
                                print("ERROR: Bad output directory given")
                                error = True

                        elif userInputArr[1] == defaults.TINY_WEIGHTS_VAR:
                            try:
                                preferences.tiny = bool(userInputArr[2])
                            except:
                                print("ERROR: Please give an true or false")
                                error = True

                        elif userInputArr[1] == defaults.TRANSFER_VAR:
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

                        elif userInputArr[1] == defaults.VALID_IN_VAR:
                            if path.exists(userInputArr[2]):
                                preferences.validate_input = userInputArr[2]
                            else:
                                print("ERROR: Failed to find directory for validation")
                                error = True

                        elif userInputArr[1] == defaults.WEIGHTS_NUM_VAR:
                            try:
                                preferences.weighted_classes = int(userInputArr[2])
                            except:
                                print("ERROR: Please give an integer value")
                                error = True

                        elif userInputArr[1] == defaults.WEIGHTS_PATH_VAR:
                            old_weights = preferences.weights_file
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
                    print_to_terminal.modify_commands()

                else:
                    print("Not enough arguments, please provide a variable and a value ie batch_size 3")

            elif userInput == "quit" or userInput == "q":
                running = False
            else:
                # end of cases, inform the user that their input was invalid
                print("\nCommand not recognized, try 'help' or 'h' for a list of options")
        except KeyboardInterrupt:
            print("\n\n\n\n------ Current process stopped by user ------")
            print("\nEnter 'help' or 'h' for a list of commands:")
            running = True

main()
