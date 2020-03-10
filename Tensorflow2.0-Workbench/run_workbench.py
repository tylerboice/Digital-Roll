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
    output_file = file_utils.get_output_file(preferences.output)
    SPLIT_CHAR = "="
    NONE = ""
    START = 1001
    CONTINUE = 1002
    SINGLE = 1003
    TEST_IMAGE = 1004
    SPECIAL_CHAR = "?<#>@"

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
    print("\t\t        - Visual Studio for C++ is installed on your machine (GPU Only)")
    print("\n\t\tAfter ensuring necessary files are in your directory and re-run the workbench\n\n")
    exit()

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def print_to_file(string):
    with open(output_file, "a") as f:
        f.write(string)

# creates and deletes a files to ensure program has admin privileges
def check_admin():
    new_file = os.getcwd() + "/test_file.txt"
    try:
       with open(new_file, "w") as f:
           f.write("testing file")
       os.remove(new_file)
    except:
       print("\n\n\tERROR: workbench needs admin privileges to modify and remove files")
       print("\n\t\tEnsure that:")
       print("\t\t        - You close the anaconda prompt and re-run the anaconda prompt as admin")
       exit()


# Prints Error message
def err_message(string):
    print_to_terminal.print_both("\n\tERROR: " + string + "\n", output_file)


# get input
def get_input(input):
    return input.replace(" ", "").lower()

def load(pref_path):
    failed = []
    changed = []
    if ".txt" not in pref_path:
        pref_path += ".txt"
    if not path.exists(pref_path):
        pref_path = os.getcwd() + "/" + pref_path
    if not path.exists(pref_path):
        err_message("Bad Preferences File, could not find file")
        return
    # Set new preferences
    preferences.pref_file = pref_path
    with open(pref_path, "r") as f:
        for line in f.readlines():
            txt_input = file_utils.get_input(line, SPLIT_CHAR)
            if defaults.BATCH_SIZE_VAR + SPLIT_CHAR in line:
                if txt_input.isnumeric():
                    if preferences.batch_size != int(txt_input):
                        preferences.batch_size = int(txt_input)
                        changed.append(defaults.BATCH_SIZE_VAR + SPECIAL_CHAR + str(preferences.batch_size))
                else:
                    failed.append(defaults.BATCH_SIZE_VAR)

            elif defaults.CLASSIFIERS_VAR + SPLIT_CHAR in line:

                if os.path.exists(txt_input) and ".names" in txt_input:
                    if preferences.classifier_file != txt_input:
                        preferences.classifier_file = txt_input
                        changed.append(defaults.CLASSIFIERS_VAR + SPECIAL_CHAR + preferences.classifier_file)

                elif os.path.exists(txt_input + ".names"):
                    if preferences.classifier_file != txt_input + ".names":
                        preferences.classifier_file = txt_input + ".names"
                        changed.append(defaults.CLASSIFIERS_VAR + SPECIAL_CHAR + preferences.classifier_file)
                else:
                    failed.append(defaults.CLASSIFIERS_VAR)

            elif defaults.DATASET_TEST_VAR + SPLIT_CHAR in line:
                if path.exists(txt_input) and ".tfrecord" in txt_input:
                    if preferences.dataset_test != txt_input:
                        preferences.dataset_test = txt_input
                        changed.append(defaults.DATASET_TEST_VAR + SPECIAL_CHAR + preferences.dataset_test)
                elif path.exists(txt_input + ".tfrecord"):
                    if preferences.dataset_test != txt_input + ".tfrecord":
                        preferences.dataset_test = txt_input + ".tfrecord"
                        changed.append(defaults.DATASET_TEST_VAR + SPECIAL_CHAR + preferences.dataset_test)
                else:
                    failed.append(defaults.DATASET_TEST_VAR)

            elif defaults.DATASET_TRAIN_VAR + SPLIT_CHAR in line:
                if path.exists(txt_input) and ".tfrecord" in txt_input:
                    if preferences.dataset_train != txt_input:
                        preferences.dataset_train = txt_input
                        changed.append(defaults.DATASET_TRAIN_VAR + SPECIAL_CHAR + preferences.dataset_train)

                elif path.exists(txt_input +  ".tfrecord"):
                    if preferences.dataset_train != txt_input + ".tfrecord":
                        preferences.dataset_train = txt_input + ".tfrecord"
                        changed.append(defaults.DATASET_TRAIN_VAR + SPECIAL_CHAR + preferences.dataset_train)
                else:
                    failed.append(defaults.DATASET_TRAIN_VAR)

            elif defaults.EPOCH_NUM_VAR + SPLIT_CHAR in line:
                if txt_input.isnumeric():
                    if preferences.epochs != int(txt_input):
                        preferences.epochs = int(txt_input)
                        changed.append(defaults.EPOCH_NUM_VAR + SPECIAL_CHAR + str(preferences.epochs))
                else:
                    failed.append(defaults.EPOCH_NUM_VAR)

            elif defaults.IMAGE_SIZE_VAR + SPLIT_CHAR in line:
                if txt_input.isnumeric():
                    if preferences.image_size != int(txt_input):
                        preferences.image_size = int(txt_input)
                        changed.append(defaults.IMAGE_SIZE_VAR + SPECIAL_CHAR + str(preferences.image_size))
                else:
                    failed.append(defaults.IMAGE_SIZE_VAR)

            elif defaults.MAX_CHECK_VAR + SPLIT_CHAR in line:
                if txt_input.isnumeric():
                    if preferences.max_checkpoints != int(txt_input):
                        preferences.max_checkpoints = int(txt_input)
                        changed.append(defaults.MAX_CHECK_VAR + SPECIAL_CHAR + str(preferences.max_checkpoints))
                else:
                    failed.append(defaults.MAX_CHECK_VAR)

            elif defaults.MAX_SESS_VAR + SPLIT_CHAR in line:
                if txt_input.isnumeric():
                    if preferences.max_saved_sess != int(txt_input):
                        preferences.max_saved_sess = int(txt_input)
                        changed.append(defaults.MAX_SESS_VAR + SPECIAL_CHAR + str(preferences.max_saved_sess))
                else:
                    failed.append(defaults.MAX_SESS_VAR)

            elif defaults.MODE_VAR + SPLIT_CHAR in line:
                if txt_input in defaults.MODE_OPTIONS:
                    if preferences.mode != txt_input:
                        preferences.mode = txt_input
                        changed.append(defaults.MODE_VAR + SPECIAL_CHAR + preferences.mode)
                else:
                    failed.append(defaults.MODE_VAR)

            elif defaults.OUTPUT_VAR + SPLIT_CHAR in line:
                if path.exists(txt_input) or "." not in txt_input:
                    if preferences.output != txt_input:
                        preferences.output = txt_input
                        changed.append(defaults.OUTPUT_VAR + SPECIAL_CHAR + preferences.output)
                else:
                    failed.append(defaults.OUTPUT_VAR)

            elif defaults.SAVED_SESS_VAR + SPLIT_CHAR in line:
                if path.exists(txt_input) or "." not in txt_input:
                    if preferences.sessions != txt_input:
                        preferences.sessions = txt_input
                        changed.append(defaults.SAVED_SESS_VAR + SPECIAL_CHAR + preferences.sessions)
                else:
                    failed.append(defaults.SAVED_SESS_VAR)

            elif defaults.TINY_WEIGHTS_VAR + SPLIT_CHAR in line:
                txt_input = txt_input.lower()
                txt_input = txt_input[0].upper() + txt_input[1:]
                if txt_input == "True" or txt_input == "False":
                    if txt_input == "True":
                        result = True
                    else:
                        result = False
                    if preferences.tiny != result:
                        preferences.tiny = result
                        changed.append(defaults.TINY_WEIGHTS_VAR + SPECIAL_CHAR + txt_input)
                else:
                    failed.append(defaults.TINY_WEIGHTS_VAR)

            elif defaults.TRANSFER_VAR + SPLIT_CHAR in line:
                if txt_input in defaults.TRANSFER_OPTIONS:
                    if preferences.transfer != txt_input:
                        preferences.transfer = txt_input
                        changed.append(defaults.TRANSFER_VAR + SPECIAL_CHAR + preferences.transfer)
                else:
                    failed.append(defaults.TRANSFER_VAR)

            elif defaults.VALID_IMGS_VAR + SPLIT_CHAR in line:
                if txt_input.isnumeric():
                    if preferences.validate_img_num != int(txt_input):
                        preferences.validate_img_num = int(txt_input)
                        changed.append(defaults.VALID_IMGS_VAR + SPECIAL_CHAR + str(preferences.validate_img_num))
                else:
                    failed.append(defaults.VALID_IMGS_VAR)

            elif defaults.VALID_IN_VAR + SPLIT_CHAR in line:
                if path.exists(txt_input):
                    if preferences.validate_input != txt_input:
                        preferences.validate_input = txt_input
                        changed.append(defaults.VALID_IN_VAR + SPECIAL_CHAR + preferences.validate_input)
                else:
                    failed.append(defaults.VALID_IN_VAR)

            elif defaults.WEIGHTS_PATH_VAR + SPLIT_CHAR in line:
                if path.exists(txt_input) and (".tf" in txt_input or ".weights" in txt_input):
                    if preferences.weights != txt_input:
                        preferences.weights = txt_input
                        changed.append(defaults.WEIGHTS_PATH_VAR + SPECIAL_CHAR + preferences.weights)

                elif path.exists(txt_input + ".tf"):
                    if preferences.weights != txt_input + ".tf":
                        preferences.weights = txt_input + ".tf"
                        changed.append(defaults.WEIGHTS_PATH_VAR + SPECIAL_CHAR + preferences.weights)

                elif path.exists(txt_input + ".weights"):
                    if preferences.weights != txt_input + ".weights":
                        preferences.weights = txt_input + ".weights"
                        changed.append(defaults.WEIGHTS_PATH_VAR + SPECIAL_CHAR + preferences.weights)

                else:
                    failed.append(defaults.WEIGHTS_PATH_VAR)

    if len(changed) != 0:
        print_to_terminal.print_both("\n\tValues changed:", output_file)
        for item in changed:
            item = item.split(SPECIAL_CHAR)
            print_to_terminal.print_both("\t\t -" + item[0] + " to " + item[1])
        print_to_terminal.print_both("\t\tUse the display(d) command to current preferences", output_file)
    else:
        print_to_terminal.print_both("\n\tNo values were altered", output_file)
    if len(failed) != 0:
        print_to_terminal.print_both("\n\n\tWARNING: The following items didn't change due incompatible input or incorrect file path:", output_file)
        for item in failed:
            print_to_terminal.print_both("\t\t -" + item)
        print_to_terminal.print_both("\t\tUse the modify(m) command with no arguments to see accepted types", output_file)



def save(save_path):
    files = 1
    changed_name = False
    if "." in save_path:
        save_path = save_path.split(".")[0]
    new_file = save_path + ".txt"
    while os.path.exists(new_file):
        new_file = save_path + "-" + str(files) + ".txt"
        changed_name = True
        files += 1
    if changed_name:
        print_to_terminal.print_both("\n\tFile " + save_path + " is already a file, using " + new_file + " instead", output_file)
    with open(new_file, "w") as f:
        f.write(defaults.BATCH_SIZE_VAR + "= " + str(preferences.batch_size) + "\n")
        f.write(defaults.CLASSIFIERS_VAR + "= " + str(preferences.classifier_file) + "\n")
        f.write(defaults.DATASET_TEST_VAR + "= " + str(preferences.dataset_test) + "\n")
        f.write(defaults.DATASET_TRAIN_VAR + "= " + str(preferences.dataset_train) + "\n")
        f.write(defaults.EPOCH_NUM_VAR + "= " + str(preferences.epochs) + "\n")
        f.write(defaults.IMAGE_SIZE_VAR + "= " + str(preferences.image_size) + "\n")
        f.write(defaults.MAX_CHECK_VAR + "= " + str(preferences.max_checkpoints) + "\n")
        f.write(defaults.MAX_SESS_VAR + "= " + str(preferences.max_saved_sess) + "\n")
        f.write(defaults.MODE_VAR + "= " + str(preferences.mode) + "\n")
        f.write(defaults.OUTPUT_VAR + "= " + str(preferences.output) + "\n")
        f.write(defaults.SAVED_SESS_VAR + "= " + str(preferences.sessions) + "\n")
        f.write(defaults.TINY_WEIGHTS_VAR + "= " + str(preferences.tiny) + "\n")
        f.write(defaults.TRANSFER_VAR + "= " + str(preferences.transfer) + "\n")
        f.write(defaults.VALID_IMGS_VAR + "= " + str(preferences.validate_img_num) + "\n")
        f.write(defaults.VALID_IN_VAR + "= " + str(preferences.validate_input) + "\n")
        f.write(defaults.WEIGHTS_PATH_VAR + "= " + str(preferences.weights) + "\n")

    print_to_terminal.print_both("\n\tNew preference path " + file_utils.from_workbench(save_path) +" successfully saved!", output_file)


def run(start_from, start_path):
    sys.stdout = Logger()
    # check if necessary files exist
    # run was called, start from beginning
    # Setting for memory growth from old train_workbench.py
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if start_from == START:
        total_images = file_utils.checkIfNecessaryPathsAndFilesExist(defaults.IMAGES_PATH,
                                                                     defaults.DATA_PATH,
                                                                     defaults.MIN_IMAGES,
                                                                     preferences.output,
                                                                     preferences.sessions,
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
        convert_weights.run_weight_convert(preferences.weights,
                                           preferences.output + "/yolov3.tf",
                                           preferences.tiny,
                                           preferences.weight_num_classes)
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
            print("\nBegin Training...")
            print("\n\t Training from scratch " + weights)
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

    if (start_from == CONTINUE or start_from == START):
        if not file_utils.is_valid(preferences.output):
            err_message(preferences.output + " not found or is empty")
            return

        if not file_utils.is_valid(preferences.classifier_file):
            err_message(preferences.classifier_file + " not found or is empty")
            return

        # update checkpoint file
        chkpnt_weights = file_utils.get_last_checkpoint(preferences.output)
        chkpnt_weights = (chkpnt_weights.split(".tf")[0] + ".tf").replace("//", "/")

        if chkpnt_weights == file_utils.ERROR or \
             file_utils.CHECKPOINT_KEYWORD not in chkpnt_weights:
            err_message("No valid checkpoints found in " + file_utils.from_workbench(preferences.output))
            print("\t\tPlease use a trained checkpoint (e.g " + file_utils.CHECKPOINT_KEYWORD + "1.tf )")
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
            temp_folder = file_utils.duplicate_pb(preferences.output)

            create_coreml.export_coreml(temp_folder)

            file_utils.remove_temp(preferences.output, temp_folder)
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
        out_img = (preferences.output + "/").replace("//", "/") + test_img.split(".")[0] + "-output.jpg"
        detect_img.run_detect(preferences.classifier_file,
                              chkpnt_weights,
                              preferences.tiny,
                              preferences.image_size,
                              test_img,
                              out_img,
                              preferences.num_classes)
    else:
        test_img = (test_img + "/").replace("//", "/")
        for file in os.listdir(test_img):
            if '.jpg' in file:
                out_img = (preferences.output + "/").replace("//", "/") + file.split(".")[0]  +"-output.jpg"
                print(out_img)
                detect_img.run_detect(preferences.classifier_file,
                                      chkpnt_weights,
                                      preferences.tiny,
                                      preferences.image_size,
                                      test_img + file,
                                      out_img,
                                      preferences.num_classes)

    if (start_from != TEST_IMAGE):
        print("\n\n=============================== Workbench Successful! ===============================")
    print("\n\tAll models and images saved in " + preferences.output + "\n")



############################## MAIN ##########################
def main():
    check_admin()
    sys.stdout = Logger()
    print("\nEnter 'help' or 'h' for a list of commands:")
    while True:
        try:
            try:
                userInput = input("\n<WORKBENCH>: ")

            except EOFError:
                print_to_terminal.print_both("\n\n\n\n------ Current process stopped by user ------", output_file)
                print("\nEnter 'help' or 'h' for a list of commands:")
                running = True

            print_to_file("\n\n===============================================\nUser Input: " +
                           userInput + "\n===============================================\n")
            if get_input(userInput) == "help" or get_input(userInput) == "h":
                print_to_terminal.help()

            elif get_input(userInput) == "run" or get_input(userInput) == "r":
                print_to_file("Running Workbench with :\n" + print_to_terminal.current_pref())
                run(START, NONE)

            elif get_input(userInput) == "lite" or get_input(userInput) == "l":
                # convert model to tensorflow lite for android use
                try:
                    # convert model to tensorflow lite for android use
                    print_to_terminal.print_both("Model Loading", output_file)
                    converter = tf.lite.TFLiteConverter.from_saved_model(preferences.output)
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                                           tf.lite.OpsSet.SELECT_TF_OPS]
                    # converter.experimental_new_converter = True
                    converter.allow_custom_ops = False  # TFLite does not support custom operations,
                    # thus this be false, to have a model with nms set to True
                    tflite_model = converter.convert()
                    open(preferences.output + "tflite_model.tflite", "wb").write(tflite_model)
                    print_to_terminal.print_both("\n\tTensorflow Lite model created!", output_file)

                except Exception as e:
                    err_message("Failed to create TF lite model: " + str(e))

            elif userInput[0:5] == "test " or userInput[0:2] == "t ":
                if userInput[0:2] == "t ":
                    img_path = userInput[2:]
                else:
                    img_path = userInput[5:]
                img_path.strip("\n\r")
                run(TEST_IMAGE, img_path)

            elif get_input(userInput) == "continue" or get_input(userInput) == "c":
                print_to_file("Coninuing Workbench with :\n" + print_to_terminal.current_pref())
                run(CONTINUE, NONE)

            elif userInput[0:5] == "continue " or userInput[0:2] == "c ":
                if userInput[0:2] == "c ":
                    prev_check = userInput[2:]
                else:
                    prev_check = userInput[5:]
                print_to_file("Coninuing Workbench with :\n" + print_to_terminal.current_pref())
                run(CONTINUE, prev_check)

            elif get_input(userInput) == "display" or get_input(userInput) == "d":
                print("\nCurrent Preferences:\n")
                print(print_to_terminal.current_pref())

            elif get_input(userInput) == "info" or get_input(userInput) == "i":
                print_to_terminal.info()


            elif userInput[0:5] == "load " or userInput[0:2] == "l ":
                if userInput[0:2] == "l ":
                    pref_path = userInput[2:]
                else:
                    pref_path = userInput[5:]
                pref_path.strip("\n")
                load(pref_path)

            elif get_input(userInput) == "save" or get_input(userInput) == "s":
                files = 1
                new_files = os.getcwd() + "/preferences-" + str(files) + ".txt"
                while os.path.exists(new_files):
                    files += 1
                    new_files = os.getcwd() + "/preferences-"  + str(files) + ".txt"
                save(new_files)

            elif userInput[0:5] == "save " or userInput[0:2] == "s ":
                if userInput[0:2] == "s ":
                    save_path = userInput[2:]
                else:
                    save_path = userInput[5:]
                save_path.strip("\n\r")
                save(save_path)

            elif get_input(userInput) == "modify" or get_input(userInput) == "m":
                print_to_terminal.modify_commands()

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
                                print_to_terminal.print_both("\n\t       ==> fit, eager_fit, eager_tf", output_file)
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
                                print_to_terminal.print_both("\t\n       ==> none, darknet, no_output, frozen, fine_tune", output_file)
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
                        print_to_terminal.print_both("\n\tSet " + userInputArr[1] + " to " + userInputArr[2] + "\n", output_file)
                elif len(userInputArr) == 2 and userInputArr[1] == '?':
                    print_to_terminal.modify_commands()

                else:
                    print_to_terminal.print_both("Not enough arguments, please provide a variable and a value ie batch_size 3", output_file)

            elif get_input(userInput) == "quit" or get_input(userInput) == "q":
                print_to_terminal.print_both("\n\tExiting workbench...", output_file)
                exit()
            else:
                # end of cases, inform the user that their input was in
                print_to_file("\nInvalid Command")
                print("\nCommand not recognized, try 'help' or 'h' for a list of options")
        except KeyboardInterrupt:
            print_to_terminal.print_both("\n\n\n\n------ Current process stopped by user ------", output_file)
            print("\nEnter 'help' or 'h' for a list of commands:")
            running = True
main()
