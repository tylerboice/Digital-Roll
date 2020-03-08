import os
import sys
from os import path

try:
    from scripts import defaults
    from scripts import file_utils
    from scripts import print_to_terminal
    from scripts import preferences
    from scripts import generate_tf
    from scripts import convert_weights
    from scripts import train_workbench
    from scripts import create_tf_model
    from scripts import detect_img
    from scripts import create_coreml
    import tensorflow as tf

    test_checkpoint = file_utils.get_last_checkpoint(preferences.output)
    SPLIT_CHAR = "="
    NONE = ""
    START = 1001
    CONTINUE = 1002
    SINGLE = 1003
    TEST_IMAGE = 1004

# if script not found
except FileNotFoundError:
    print("\n\n\tERROR: files needed for the workbench to run were not found")
    print("\n\t\tEnsure that:")
    print("\t\t        - The scripts folder and yolov3_tf2 folder have not been removed or altered")
    print("\t\t        - You are in the proper directory")
    print("\n\t\tAfter ensuring necessary files are in your directory and re-run the workbench\n\n")
    exit()

except ImportError:
    print("\n\n\tERROR: packages needed for the workbench to run were not found")
    print("\n\t\tEnsure that:")
    print("\t\t        - Your conda enviorment is activated")
    print("\t\t        - You have installed the proper packages using the requirements.txt file")
    print("\t\t        - Visual Studio for C++ is installed on your machine")
    print("\n\t\tAfter ensuring necessary files are in your directory and re-run the workbench\n\n")
    exit()

# Prints Error message
def err_message(string):
    print("\n\tERROR: " + string + "\n")


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def load(pref_path):
    if path.exists(pref_path):
        # Set new preferences
        preferences.pref_file = pref_path
        with open(pref_path, "r") as f:
            for line in f.readlines():
                txt_input = file_utils.get_input(line, SPLIT_CHAR)
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
                        preferences.num_classes = file_utils.get_num_classes(preferences.classifier_file)
                    except:
                        try:
                            preferences.num_classes = file_utils.get_num_classes(os.getcwd().replace("\\", "/")
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
                            or txt_input == "no_output" \
                            or txt_input == "frozen" \
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
                            preferences.weight_num_classes = file_utils.get_num_classes(
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


def run(start_from, start_path):
    single_script = False
    # check if necessary files exist
    # run was called, start from beginning

    if start_from == START:
        total_images = file_utils.checkIfNecessaryPathsAndFilesExist(defaults.IMAGES_PATH,
                                                                     defaults.DATA_PATH,
                                                                     defaults.MIN_IMAGES,
                                                                     preferences.output,
                                                                     defaults.TEST_IMAGE_PATH,
                                                                     defaults.TRAIN_IMAGE_PATH,
                                                                     defaults.VALIDATE_IMAGE_PATH,
                                                                     defaults.YOLO_PATH)

        if total_images == 0:
            err_message("No images have been found in the image folder")
            print("\t\tImage Folder Location: " + defaults.IMAGES_PATH)
            print(
                "\n\t\tFor an example set, look at the Pre_Labeled_Images folder in the repository or at https://github.com/tylerboice/Digital-Roll\n")
            return

        elif total_images < defaults.MIN_IMAGES:
            err_message("Workbench needs at minimum " + str(defaults.MIN_IMAGES) + " to train. Current Count: " + str(total_images))
            print("\t       However it is recommended you have around 1000 per classifier")
            return

        # create classifiers.names
        print("\nTraining Data Info:")
        print("\n\tTotal images = " + str(total_images))

        all_classifiers = file_utils.get_classifiers(defaults.IMAGES_PATH)
        classifiers = file_utils.classifier_remove_dup(all_classifiers)
        all_classifiers = []
        file_utils.create_classifier_file(preferences.classifier_file, classifiers)


        # sort all the images
        print("\nSorting images...")
        file_utils.sort_images(preferences.validate_img_num,
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

        if not file_utils.is_valid(preferences.dataset_test) or not file_utils.is_valid(preferences.dataset_train):
            err_message("Not enough image given for the workbench, or the data is not properly set-up")
            print("\t\tMake sure every .xml has a corresponing image and you have at least " + str(
                defaults.MIN_IMAGES) + " images")
            exit()
        print("\n\tSuccessfully generated tf records!\n")

        # save previous sessions
        print("\nChecking for previous Sessions...\n")
        file_utils.save_session(defaults.OUTPUT_PATH, preferences.output, defaults.SAVED_SESS_PATH,
                                preferences.max_saved_sess)
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

    # if training
    if (((start_from) == CONTINUE and start_path != NONE) or start_from == START):

        # continue training from previous checkpoint
        if (start_from != START):
            weights = start_path
            if os.path.isdir(weights):
                weights = file_utils.get_last_checkpoint(weights)
                weights = (weights.split(".tf")[0] + ".tf").replace("//", "/")
            if ".tf" not in weights:
                err_message("File is not a checkpoint")
                print("\n\t\tCheckpoint Example: yolov3_train_3.tf")
                return
            if "ERROR" in weights:
                err_message("No checkpoints found in " + start_path)
                return
            print("\n\tContinuing from " + weights)
            print("\nResume Training...")
            transfer_mode = 'fine_tune'

        # train from scratch
        else:
            print("\n\t Training for scratch " + weights)
            print("\nBegin Training...")
            transfer_mode = preferences.transfer

        # start training
        print("\n\tThis will take some time...\n")
        trained = train_workbench.run_train(preferences.dataset_train,
                                            preferences.dataset_test,
                                            preferences.tiny,
                                            defaults.IMAGES_PATH,
                                            weights,
                                            preferences.classifier_file,
                                            preferences.mode,
                                            transfer_mode,
                                            preferences.image_size,
                                            preferences.epochs,
                                            preferences.batch_size,
                                            defaults.DEFAULT_LEARN_RATE,
                                            preferences.num_classes,
                                            preferences.weight_num_classes,
                                            preferences.output,
                                            preferences.max_checkpoints)
        if not trained:
            return
        print("\n\tTraining Complete!\n")

    if (start_from != TEST_IMAGE):
        if not file_utils.is_valid(preferences.output):
            err_message(preferences.output + " not found or is empty")
            return

        if not file_utils.is_valid(preferences.classifier_file):
            err_message(preferences.classifier_file + " not found or is empty")
            return

        # update checkpoint file
        chkpnt_weights = file_utils.get_last_checkpoint(preferences.output)
        chkpnt_weights = (chkpnt_weights.split(".tf")[0] + ".tf").replace("//", "/")

        if chkpnt_weights == file_utils.ERROR:
            print("\n\tNo valid checkpoints found in " + file_utils.from_workbench(preferences.output))
            return
        elif file_utils.CHECKPOINT_KEYWORD not in chkpnt_weights:
            print("\tPlease use a trained checkpoint (e.g " + file_utils.CHECKPOINT_KEYWORD + "1.tf )")
            return

        file_utils.rename_checkpoints(preferences.output, preferences.max_checkpoints)
        file_utils.write_to_checkpoint(chkpnt_weights, (preferences.output + "/checkpoint").replace("//", "/"))

        print("\n\tUsing checkpoint " + file_utils.from_workbench(chkpnt_weights) )

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


        # create Tensorflow Lite model
        try:
            # convert model to tensorflow lite for android use
            print("\nModel Loading...")
            converter = tf.lite.TFLiteConverter.from_saved_model(preferences.output)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                                   tf.lite.OpsSet.SELECT_TF_OPS]

            converter.allow_custom_ops = True  # TFLite does not support custom operations,
                                                # thus this be false, to have a model with nms set to True
            tflite_model = converter.convert()
            open(preferences.output + "tflite_model.tflite", "wb").write(tflite_model)
            print("\n\tTensorflow Lite model created!\n")

        except Exception as e:
            err_message("Failed to create TF lite model: " + str(e))

        # Create Core ML Model
        try:
            print("\nCreating a CoreML model...")
            blockPrint()
            create_coreml.export_coreml(preferences.output)
            enablePrint()
            print("\n\tCore ML model created!\n")

        except Exception as e:
            err_message("Failed to create CoreML model: " + str(e))

    # generating tensorflow models
    if (start_from == TEST_IMAGE):
        test_img = start_path

    else:
        test_img = preferences.validate_input

    chkpnt_weights = (file_utils.get_last_checkpoint(preferences.output))
    chkpnt_weights = (chkpnt_weights.split(".tf")[0] + ".tf").replace("//", "/")
    if chkpnt_weights == file_utils.ERROR:
        err_message("No checkpoints found in " + start_path)
        return

    if not os.path.isdir(test_img) and file_utils.is_valid(test_img):
        print("\n\tTest image location not found " + test_img)
        return

    print("\nTesting Images...")
    if path.isfile(test_img):
        detect_img.run_detect(preferences.classifier_file,
                              chkpnt_weights,
                              preferences.tiny,
                              preferences.image_size,
                              test_img,
                              preferences.output,
                              preferences.num_classes)
    else:
        test_img = (test_img + "/").replace("//", "/")
        for file in os.listdir(test_img):
            if '.jpg' in file:
                detect_img.run_detect(preferences.classifier_file,
                                      chkpnt_weights,
                                      preferences.tiny,
                                      preferences.image_size,
                                      test_img + file,
                                      preferences.output + file + "_output.jpg",
                                      preferences.num_classes)

    if (start_from != TEST_IMAGE):
        print("\n\n=============================== Workbench Successful! ===============================")
    print("\n\tAll models and images saved in " + preferences.output + "\n")



############################## MAIN ##########################
def main():
    print("\nWelcome to the Digital Roll Workbench")
    print("\nEnter 'help' or 'h' for a list of commands:")
    running = True
    while running:
        enablePrint()
        try:
            try:
                userInput = input("\n<WORKBENCH>: ")

            except EOFError:
                print("\n\n\n\n------ Current process stopped by user ------")
                print("\nEnter 'help' or 'h' for a list of commands:")
                running = True
            userInput.lower()
            userInput.strip()

            if userInput.replace(" ", "") == "help" or userInput.replace(" ", "") == "h":
                print_to_terminal.help()

            elif userInput.replace(" ", "") == "run" or userInput.replace(" ", "") == "r":
                run(START, NONE)

            elif userInput.replace(" ", "") == "lite" or userInput.replace(" ", "") == "l":
                # convert model to tensorflow lite for android use
                try:
                    # convert model to tensorflow lite for android use
                    print("Model Loading")
                    converter = tf.lite.TFLiteConverter.from_saved_model(preferences.output)
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                                           tf.lite.OpsSet.SELECT_TF_OPS]
                    # converter.experimental_new_converter = True
                    converter.allow_custom_ops = False  # TFLite does not support custom operations,
                    # thus this be false, to have a model with nms set to True
                    tflite_model = converter.convert()
                    open(preferences.output + "tflite_model.tflite", "wb").write(tflite_model)
                    print("\n\tTensorflow Lite model created!")

                except Exception as e:
                    err_message("Failed to create TF lite model: " + str(e))

            elif userInput[0:5] == "test " or userInput[0:2] == "t ":
                error = False
                if userInput[0:2] == "t ":
                    img_path = userInput[2:]
                else:
                    img_path = userInput[5:]
                img_path.strip("\n\r")
                run(TEST_IMAGE, img_path)

            elif userInput.replace(" ", "") == "continue" or userInput.replace(" ", "") == "c":
                run(CONTINUE, NONE)

            elif userInput[0:5] == "continue " or userInput[0:2] == "c ":
                if userInput[0:2] == "c ":
                    prev_check = userInput[2:]
                else:
                    prev_check = userInput[5:]
                run(CONTINUE, prev_check)

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
                                preferences.num_classes = file_utils.get_num_classes(os.getcwd().replace("\\", "/")
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
                                preferences.weight_num_classes = file_utils.get_num_classes(
                                    os.getcwd().replace("\\", "/")
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
                exit()
            else:
                # end of cases, inform the user that their input was invalid
                print("\nCommand not recognized, try 'help' or 'h' for a list of options")
        except KeyboardInterrupt:
            enablePrint()
            print("\n\n\n\n------ Current process stopped by user ------")
            print("\nEnter 'help' or 'h' for a list of commands:")
            running = True
main()
