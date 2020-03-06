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
import tensorflow as tf

test_checkpoint = files.get_last_checkpoint(preferences.output)
SPLIT_CHAR = "="
START = 1001
CONTINUE = 1002
SINGLE = 1003

# Prints Error message
def err_message(string):
    print("\n\tERROR: " + string + "\n")

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
        files.save_checkpoints(preferences.output, preferences.sessions, preferences.max_saved_sess)
        convert_weights.run_weight_convert(preferences.weights,
                                           preferences.output,
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
                                  default.DEFAULT_LEARN_RATE,
                                  preferences.num_classes,
                                  preferences.weight_num_classes,
                                  preferences.output,
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
                        err_message("Bad batch size given, cannot convert value to int")
                        error = True

                elif defaults.TEST_CHECKPOINT_VAR + SPLIT_CHAR in line:
                    try:
                        this.test_checkpoint = txt_input
                    except:
                        err_message("Bad testing checkpoint directory given")
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
                            err_message("Failed to update classifier file, new file not found")
                            preferences.classifier_file = old_classifier
                            error = True

                elif defaults.DATASET_TEST_VAR + SPLIT_CHAR in line:
                    if path.exists(txt_input):
                        preferences.dataset_test = txt_input
                    else:
                        err_message("Bad test dataset directory given")
                        error = True

                elif defaults.DATASET_TRAIN_VAR + SPLIT_CHAR in line:
                    if path.exists(txt_input):
                        preferences.dataset_train = txt_input
                    else:
                        err_message("Bad train dataset directory given")
                        error = True

                elif defaults.EPOCH_NUM_VAR + SPLIT_CHAR in line:
                    try:
                        preferences.epochs = int(txt_input)
                    except:
                        err_message("Bad epochs value given, cannot convert to int")
                        error = True

                elif defaults.IMAGE_SIZE_VAR + SPLIT_CHAR in line:
                    try:
                        preferences.image_size = int(txt_input)
                    except:
                        err_message("Bad image size value given, cannot convert to int")
                        error = True

                elif defaults.MAX_CHECK_VAR + SPLIT_CHAR in line:
                    try:
                        preferences.max_checkpoints = int(txt_input)
                    except:
                        err_message("Bad max check value given, cannot convert to int")
                        error = True

                elif defaults.MAX_SAVED_SESS_VAR + SPLIT_CHAR in line:
                    try:
                        preferences.max_saved_sess = int(txt_input)
                    except:
                        err_message("Bad max saved sessions value given, cannot convert to int")
                        error = True

                elif defaults.MODE_VAR + SPLIT_CHAR in line:
                    if txt_input == "fit" \
                            or txt_input == "eager_fit" \
                            or txt_input == "eager_tf":
                        preferences.mode = txt_input
                    else:
                        err_message("Bad mode value given, please update the file and choose one of the following")
                        print("\t\n       ==> fit, eager_fit, eager_tf")
                        error = True

                elif defaults.OUTPUT_VAR + SPLIT_CHAR in line:
                    if path.exists(txt_input):
                        preferences.output = txt_input
                    else:
                        err_message("Bad output directory given")
                        error = True

                elif defaults.TINY_WEIGHTS_VAR + SPLIT_CHAR in line:
                    try:
                        preferences.tiny = bool(txt_input)
                    except:
                        err_message("Failed to give True/False to tiny value")
                        error = True

                elif defaults.TRANSFER_VAR + SPLIT_CHAR in line:
                    if txt_input == "none" \
                            or txt_input == "darknet" \
                            or txt_input == "no_output"\
                            or txt_input == "frozen"\
                            or txt_input == "fine_tune":
                        preferences.transfer = txt_input
                    else:
                        err_message("Bad transfer value given, please update the file and choose one of the following")
                        print("\n\t       ==> none, darknet, no_output, frozen, fine_tune")
                        error = True

                elif defaults.VALID_IMGS_VAR + SPLIT_CHAR in line:
                    try:
                        preferences.validate_img_num = int(txt_input)
                    except:
                        err_message("Bad valid image value given, cannot convert to int")
                        error = True

                elif defaults.VALID_IN_VAR + SPLIT_CHAR in line:
                    txt_input = line.split("=")[1]
                    txt_input = txt_input.strip()
                    if path.exists(txt_input):
                        preferences.validate_input = txt_input
                    else:
                        err_message("Failed to find directory for validation")
                        error = True

                elif defaults.WEIGHTS_PATH_VAR + SPLIT_CHAR in line:
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
                            err_message("Failed to update weights file, new file not found")
                            preferences.weights = old_weights
                            error = True
                    else:
                        err_message("Failed to update weights path")
                        error = True
        if error:
            print("A setting has failed to load properly please check above errors for info")
        else:
            print("\nNew Preferences:")
            print_to_terminal.current_pref()
    else:
        err_message("Bad Preferences File, could not find file")

def save(save_path):
    with open(save_path, "w") as f:
        f.write(defaults.BATCH_SIZE_VAR + "= " + str(preferences.batch_size) + "\n")
        f.write(defaults.TEST_CHECKPOINT_VAR + "= " + str(test_checkpoint) + "\n")
        f.write(defaults.CLASSIFIERS_VAR + "= " + str(preferences.classifier_file) + "\n")
        f.write(defaults.DATASET_TEST_VAR + "= " + str(preferences.dataset_test) + "\n")
        f.write(defaults.DATASET_TRAIN_VAR + "= " + str(preferences.dataset_train) + "\n")
        f.write(defaults.EPOCH_NUM_VAR + "= " + str(preferences.epochs) + "\n")
        f.write(defaults.IMAGE_SIZE_VAR + "= " + str(preferences.image_size) + "\n")
        f.write(defaults.MODE_VAR + "= " + str(preferences.mode) + "\n")
        f.write(defaults.MAX_CHECK_VAR + "= " + str(preferences.max_checkpoints) + "\n")
        f.write(defaults.MAX_SAVED_SESS_VAR + "= " + str(preferences.max_saved_sess) + "\n")
        f.write(defaults.OUTPUT_VAR + "= " + str(preferences.output) + "\n")
        f.write(defaults.TINY_WEIGHTS_VAR + "= " + str(preferences.tiny) + "\n")
        f.write(defaults.TRANSFER_VAR + "= " + str(preferences.transfer) + "\n")
        f.write(defaults.VALID_IMG_VAR + "= " + str(preferences.validate_img_num) + "\n")
        f.write(defaults.VALID_IN_VAR + "= " + str(preferences.validate_input) + "\n")
        f.write(defaults.WEIGHTS_CLASS_VAR + "= " + str(preferences.weight_num_classes) + "\n")


def run(start_from):

    single_script = False
    # check if necessary files exist
    # run was called, start from beginning

    if start_from == START:
        total_images = files.checkIfNecessaryPathsAndFilesExist(defaults.IMAGES_PATH,
                                                         defaults.MIN_IMAGES,
                                                         preferences.output,
                                                         defaults.TEST_IMAGE_PATH,
                                                         defaults.TRAIN_IMAGE_PATH,
                                                         defaults.VALIDATE_IMAGE_PATH,
                                                         defaults.YOLO_PATH)

        if total_images == 0:
            err_message("No images have been found in the image folder")
            print("\t\tImage Folder Location: " + defaults.IMAGES_PATH)
            print("\n\t\tFor an example set, look at the Pre_Labeled_Images folder in the repository or at https://github.com/tylerboice/Digital-Roll\n")
            return

        elif total_images < defaults.MIN_IMAGES:
            err_message("Workbench needs at minimum " + str(defaults.MIN_IMAGES) + " to train" )
            print("\t       However it is recommended you have around 1000 per classifier")
            return


        # create classifiers.names
        print("\nGathering classifier data...")
        classifiers = files.get_classifiers(defaults.IMAGES_PATH)
        files.create_classifier_file(preferences.classifier_file, classifiers)
        print("\n\tData successfuly classified!\n\n")

        # sort all the images
        print("Sorting images...")
        files.sort_images(preferences.validate_img_num,
                          defaults.IMAGES_PATH,
                          defaults.TEST_IMAGE_PATH,
                          defaults.TRAIN_IMAGE_PATH,
                          preferences.validate_input,
                          total_images
                          )
        print("\n\tAll images sorted!\n\n")

        # generate tf records
        if len(classifiers) == 0:
            err_message("No Classifiers found, make sure you labelled the images")
            return

        print("Generating images and xml files into tfrecords...\n")
        generate_tf.generate_tfrecords(defaults.TRAIN_IMAGE_PATH,
                                      preferences.dataset_train)
        generate_tf.generate_tfrecords(defaults.TEST_IMAGE_PATH,
                                      preferences.dataset_test)
        print("\n\tSuccessfully generated tf records!\n")

        # save previous sessions
        print("\nChecking for previous Sessions...\n")
        files.save_session(defaults.OUTPUT_PATH, preferences.output, defaults.SAVED_SESS_PATH, preferences.max_saved_sess)
        print("\tDone!\n")

        # convert to checkpoint
        print("\nConverting records to checkpoint...\n")
        blockPrint()
        convert_weights.run_weight_convert(preferences.weights,
                                           preferences.output + "/yolov3.tf",
                                           preferences.tiny,
                                           preferences.weight_num_classes)
        enablePrint()
        weights = (preferences.output + "/yolov3.tf").replace("//", "/")

        print("\tCheckpoint Converted!\n")


    if(start_from != CONTINUE):

        # continue training from previous checkpoint
        if(start_from != START):
            weights = start_from
            if os.path.isdir(weights):
                weights = files.get_last_checkpoint(weights)
                weights = (weights.split(".tf")[0] + ".tf").replace("//", "/")
            if ".tf" not in weights:
                err_message("File is not a  checkpoint")
                print("\n\t\tCheckpoint Example: yolov3_train_3.tf")
                return
            weight_num = preferences.num_classes
            print("\n\tContinuing from " + weights)
            print("\nResume Training... \n")
            transfer_mode = "fine_tune"

        # train from scratch
        else:
            print("\nBegin Training... \n")
            weight_num = preferences.weight_num_classes
            transfer_mode = preferences.transfer

        # start training
        train_workbench.run_train(preferences.dataset_train,
                                  preferences.dataset_test,
                                  preferences.tiny,
                                  convert_weights,
                                  preferences.classifier_file,
                                  preferences.mode,
                                  transfer_mode,
                                  preferences.image_size,
                                  preferences.epochs,
                                  preferences.batch_size,
                                  defaults.DEFAULT_LEARN_RATE,
                                  preferences.num_classes,
                                  weight_num,
                                  preferences.output,
                                  preferences.max_checkpoints )
        print("\n\tTraining Complete!\n\n")

    # update checkpoint file
    chkpnt_weights = files.get_last_checkpoint(preferences.output)
    chkpnt_weights = (chkpnt_weights.split(".tf")[0] + ".tf").replace("//", "/")

    if chkpnt_weights == files.ERROR:
        print("\n\tNo valid checkpoints found in " + files.from_workbench(preferences.output))
        return

    files.rename_checkpoints(preferences.output, preferences.max_checkpoints)
    files.write_to_checkpoint(chkpnt_weights, (preferences.output + "/checkpoint").replace("//", "/"))

    # generating tensorflow models
    print("\nGenerating TensorFlow model...\n")

    test_img = preferences.validate_input
    if os.path.isdir(test_img):
        for file in os.listdir(preferences.validate_input):
            if '.jpg' in file:
                test_img = preferences.validate_input + file
    if ".jpg" not in test_img:
        print("validate_input is not an image or does not contain an image")
        return

    create_tf_model.run_export_tfserving(chkpnt_weights,
                                          preferences.tiny,
                                          preferences.output,
                                          preferences.classifier_file,
                                          test_img,
                                          preferences.num_classes)
    print("\n\tTensorFlow model Generated!\n")



    # create Tensorflow Lite model
    try:
        # convert model to tensorflow lite for android use
        converter = tf.lite.TFLiteConverter.from_saved_model(preferences.output)
        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)

        print("\n\tTensorflow Lite model created!")

    except:
        err_message("Failed to create TF lite model")


    # Create Core ML Model
    try:
        print("\nCreating a CoreML model...\n")
        create_coreml.export_coreml(preferences.output)
        print("\n\tCore ML model created!")

    except:
        err_message("Failed to create CoreML model")



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
            if userInput.replace(" ", "") == "help" or userInput.replace(" ", "") == "h":
               print_to_terminal.help()

            elif userInput.replace(" ", "") == "run" or userInput.replace(" ", "") == "r":
                run(START)

            elif userInput.replace(" ", "") == "lite" or userInput.replace(" ", "") == "l":
                # convert model to tensorflow lite for android use
                converter = tf.lite.TFLiteConverter.from_saved_model(preferences.output)
                tflite_model = converter.convert()
                open("converted_model.tflite", "wb").write(tflite_model)

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
                    err_message("Could not find " + img_path)

            elif userInput.replace(" ", "") == "continue" or userInput.replace(" ", "") == "c":
                run(CONTINUE)

            elif userInput[0:5] == "continue " or userInput[0:2] == "c ":
                if userInput[0:2] == "c ":
                    prev_check = userInput[2:]
                else:
                    prev_check = userInput[5:]
                run(prev_check)

            elif userInput.replace(" ", "") == "display" or userInput.replace(" ", "") == "d":
                print_to_terminal.current_pref()

            elif userInput.replace(" ", "") == "info" or userInput.replace(" ", "") == "i":
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
                    err_message("Loading failed, see above errors for more info")

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
                        err_message("Failed to save")
                else:
                    err_message("File with this name already exists at this location")

            elif userInput[0:7] == "modify " or userInput[0:2] == "m ":
                error = False
                userInputArr = userInput.split(" ")
                if len(userInputArr) == 3:
                    try:
                        if userInputArr[1] == defaults.BATCH_SIZE_VAR:
                            try:
                                preferences.batch_size = int(userInputArr[2])
                            except:
                                err_message(defaults.BATCH_SIZE_VAR + " taks an integer value")
                                error = True


                        elif userInputArr[1] == "test_checkpoint":
                            if path.exists(userInputArr[2]):
                                this.test_checkpoint = userInputArr[2]
                            else:
                                err_message("Bad testing checkpoint directory given")
                                error = True

                        elif userInputArr[1] == defaults.CLASSIFIERS_VAR:
                            old_classifier = preferences.classifier_file
                            preferences.classifier_file = userInputArr[2]
                            try:
                                preferences.num_classes = files.get_num_classes(os.getcwd().replace("\\", "/")
                                                                                + "/"
                                                                                + preferences.classifier_file[1:])
                            except:
                                err_message("Failed to update classifier file, new file not found")
                                preferences.classifier_file = old_classifier
                                error = True

                        elif userInputArr[1] == defaults.DATASET_TEST_VAR:
                            if path.exists(userInputArr[2]):
                                 preferences.dataset_test = userInputArr[2]
                            else:
                                err_message("Bad dataset test directory given")
                                error = True

                        elif userInputArr[1] == defaults.DATASET_TRAIN_VAR:
                            if path.exists(userInputArr[2]):
                                 preferences.dataset_train = userInputArr[2]
                            else:
                                err_message("Bad dataset train directory given")
                                error = True

                        elif userInputArr[1] == defaults.EPOCH_NUM_VAR:
                            try:
                                preferences.epochs = int(userInputArr[2])
                            except:
                                print_to_terminal.err_message("Please give an integer value")
                                error = True

                        elif userInputArr[1] == defaults.IMAGE_SIZE_VAR:
                            try:
                                preferences.image_size = int(userInputArr[2])
                            except:
                                print_to_terminal.err_message("Please give an integer value")
                                error = True

                        elif userInputArr[1] == defaults.MAX_CHECK_VAR:
                            try:
                                preferences.max_checkpoints = int(userInputArr[2])
                            except:
                                print_to_terminal.err_message("Please give an integer value")
                                error = True

                        elif userInputArr[1] == defaults.MAX_SESS_VAR:
                            try:
                                preferences.max_sessions = int(userInputArr[2])
                            except:
                                err_message("Please give an integer value")
                                error = True

                        elif userInputArr[1] == defaults.MODE_VAR:
                            if userInputArr[2] == "fit" \
                                    or userInputArr[2] == "eager_fit" \
                                    or userInputArr[2] == "eager_tf":
                                preferences.mode = userInputArr[2]
                            else:
                                err_message("Bad mode value given, please choose one of the following")
                                print("\n\t       ==> fit, eager_fit, eager_tf")
                                error = True

                        elif userInputArr[1] == defaults.MODE_VAR:
                            if path.exists(userInputArr[2]):
                                 preferences.output = userInputArr[2]
                            else:
                                print_to_terminal.err_message("Bad output directory given")
                                error = True

                        elif userInputArr[1] == defaults.TINY_WEIGHTS_VAR:
                            try:
                                preferences.tiny = bool(userInputArr[2])
                            except:
                                print_to_terminal.err_message("Please give an true or false")
                                error = True

                        elif userInputArr[1] == defaults.TRANSFER_VAR:
                            if userInputArr[2] == "none" \
                                    or userInputArr[2] == "darknet" \
                                    or userInputArr[2] == "no_output" \
                                    or userInputArr[2] == "frozen" \
                                    or userInputArr[2] == "fine_tune":
                                preferences.transfer = userInputArr[2]
                            else:
                                err_message("Bad transfer value given, please choose one of the following")
                                print("\t\n       ==> none, darknet, no_output, frozen, fine_tune")
                                error = True

                        elif userInputArr[1] == defaults.VALID_IMGS_VAR:
                            try:
                                preferences.validate_img_num = int(userInputArr[2])
                            except:
                                err_message("Please give an integer value")
                                error = True
                                
                        elif userInputArr[1] == defaults.VALID_IN_VAR:
                            if path.exists(userInputArr[2]):
                                preferences.validate_input = userInputArr[2]
                            else:
                                err_message("Failed to find directory for validation")
                                error = True

                        elif userInputArr[1] == defaults.WEIGHTS_NUM_VAR:
                            try:
                                preferences.weighted_classes = int(userInputArr[2])
                            except:
                                err_message("Please give an integer value")
                                error = True

                        elif userInputArr[1] == defaults.WEIGHTS_PATH_VAR:
                            old_weights = preferences.weights_file
                            preferences.weights = userInputArr[2]
                            try:
                                preferences.weight_num_classes = files.get_num_classes(os.getcwd().replace("\\", "/")
                                                                                       + "/"
                                                                                       + preferences.weights[1:])
                            except:
                                err_message("Failed to update weights file, new file not found")
                                preferences.weights = old_weights
                                error = True

                        else:
                            err_message("Unknown variable name")
                            error = True
                    except:
                        err_message("Failed to change variable to given value, try 'change ?' for a list of names")
                        error = True
                    if not error:
                        print("\n\tSet " + userInputArr[1] + " to " + userInputArr[2] + "\n")
                elif len(userInputArr) == 2 and userInputArr[1] == '?':
                    print_to_terminal.modify_commands()

                else:
                    print("Not enough arguments, please provide a variable and a value ie batch_size 3")

            elif userInput.replace(" ", "") == "quit" or userInput.replace(" ", "") == "q":
                running = False
            else:
                # end of cases, inform the user that their input was invalid
                print("\nCommand not recognized, try 'help' or 'h' for a list of options")
        except KeyboardInterrupt:
            print("\n\n\n\n------ Current process stopped by user ------")
            print("\nEnter 'help' or 'h' for a list of commands:")
            running = True

main()
