# import packages
import os
import sys
from os import path
import warnings
import time

# try to import scripts
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
    from tensorflow.keras.applications import MobileNet

    from yolov3_tf2.models import (
        YoloV3, YoloV3Tiny
    )

    test_checkpoint = file_utils.get_last_checkpoint(preferences.output)
    output_file = file_utils.get_output_file()
    SPLIT_CHAR = "="
    NONE = ""
    START = 1001
    CONTINUE = 1002
    SINGLE = 1003
    TEST_IMAGE = 1004
    SPECIAL_CHAR = "?<#>@"
    INPUT_ERR = -99999999
    NO_ERROR = -99999998

# if script not found
except FileNotFoundError:
    print("\n\n\tERROR: files needed for the workbench to run were not found")
    print("\n\t\tEnsure that:")
    print("\t\t        - The scripts folder and yolov3_tf2 folder have not been removed or altered")
    print("\t\t        - You are in the proper directory")
    print("\n\t\tAfter ensuring necessary files are in your directory and re-run the workbench\n\n")
    exit()

# if package not found
except ImportError:
    print("\n\n\tERROR: packages needed for the workbench to run were not found")
    print("\n\t\tEnsure that:")
    print("\t\t        - Your conda enviorment is activated")
    print("\t\t        - You have installed the proper packages using the requirements.txt file")
    print("\t\t        - Visual Studio for C++ is installed on your machine (GPU Only)")
    print("\n\t\tAfter ensuring necessary files are in your directory and re-run the workbench\n\n")
    exit()

# Logger class to write to file and terminal
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# start logger
sys.stdout = Logger()


########################## CHECK_ADMIN #############################
# Description: creates and deletes a files to ensure program has admin privileges
# Parameters: None
# Return: Nothing
def check_admin():
    new_file = os.getcwd() + "/test_file.txt"
    try:
       with open(new_file, "w") as f:
           f.write("test.file")
       os.remove(new_file)
    except:
       print("\n\n\tERROR: workbench needs admin privileges to modify and remove files")
       print("\n\t\tEnsure that:")
       print("\t\t        - You close the anaconda prompt and re-run the anaconda prompt as admin")
       exit()


########################## ERR_MESSAGE #############################
# Description: takes a string and prints it with a "\n\tERROR: " prefix
# Parameters: string - String - string to print
# Return: Nothing
def err_message(string):
    print("\n\tERROR: " + string + "\n")


########################## GET_INPUT #############################
# Description: takes a string, removes all the spaces and sets it to lower case
# Parameters: input - String - string to remove spaces a make lower
# Return: String - input string with spaces removed and lower case
def get_input(input):
    return input.replace(" ", "").lower()


########################## CHECK_INPUT #############################
# Description: takes a value and type and ensures the value is the same type
# Parameters: value - string - a value to check if type matches type
#             type - int - a hard coded value that reprensents types uses values from
#                          default file: INT FLOAT BOOL FILE also can be a list
# Return: the converted value to the type or if value is in list(type). If Failed
#         returns INPUT_ERR if the value is not the type
def check_input(value, type):

    # integer varable
    if type == defaults.INT:
        value = value.replace(" ", "")
        if value.isnumeric():
            if int(value) <= 0:
                err_message(" Variable needs to be greater than 0")
            else:
                return int(value)
        else:
            err_message( value + " is not a integer")

    # float variable
    elif type == defaults.FLOAT:
        if value.isfloat():
            if float(value) <= 0:
                err_message( "variable needs to be greater than 0")
            else:
                return float(value)
        else:
            err_message( value + " is not a float")

    # string/file variable
    elif type == defaults.FILE:
        if path.exists(value):
            return value
        err_message( value + " does not exist")

    # boolean variable
    elif type == defaults.BOOL:
        if "true"in value.lower():
            return True
        elif "false" in value.lower():
            return False
        err_message( value + " is not boolean")

    # list variable
    else:
        value = txt_input.replace(" ", "")
        value = txt_input.replace("\n", "")
        if value in type:
            return value
        err_message( value + " is not an option")

    return INPUT_ERR


########################## Modify #############################
# Description: takes user input variable and value and changes variable to that value, if
#              possible. if not then don't change anythin and prints error
# Parameters: user_var - string - user input, name of the variable they want to change
#             user_input - string - user input, the value the variable is being changed to
# Return: Nothing
def modify(user_var, user_input):

    # check if it is varibale that should be an Integers
    if user_var == defaults.BATCH_SIZE_VAR or \
       user_var == defaults.EPOCH_NUM_VAR or \
       user_var == defaults.IMAGE_SIZE_VAR or \
       user_var == defaults.MAX_CHECK_VAR or \
       user_var == defaults.MAX_SESS_VAR or \
       user_var == defaults.VALID_IMGS_VAR or \
       user_var == defaults.WEIGHTS_NUM_VAR:

       user_input = check_input(user_input, defaults.INT)

    # check if it is varibale that should be a File
    elif user_var == defaults.CLASSIFIERS_VAR or \
         user_var == defaults.DATASET_TEST_VAR or \
         user_var == defaults.DATASET_TRAIN_VAR or \
         user_var == defaults.OUTPUT_VAR or \
         user_var == defaults.SAVED_SESS_VAR or \
         user_var == defaults.VALID_IN_VAR:

         user_input = check_input(user_input, defaults.FILE)

    # check if it is varibale that should be a Boolean
    elif user_var == defaults.TINY_WEIGHTS_VAR:
         user_input = check_input(user_input, defaults.BOOL)

    # check if it is varibale that should be a List
    elif user_var == defaults.TRANSFER_VAR or \
         user_var == defaults.MODE_VAR:
         valid_var = True

    # variable not found, return ERROR
    else:
        err_message("Variable not found, use the modifiy(m) command without an argument for list of variables")
        user_input = INPUT_ERR

    # if not error
    if user_input != INPUT_ERR:

        # batch_size - INT
        if user_var == defaults.BATCH_SIZE_VAR :
            preferences.batch_size = user_input

        # classifiers - File(string)
        elif user_var == defaults.CLASSIFIERS_VAR:
            if ".names" in user_input:
                preferences.classifier_file = user_input
                num_classes = file_utils.get_num_classes(user_input)
                preferences.num_classes = num_classes
                print("\n\tSet num_classes set to " + str(num_classes))
            else:
                err_message("Classifier file must be a .names file")
                user_input = INPUT_ERR

        # dataset_test - File(string)
        elif user_var == defaults.DATASET_TEST_VAR:
            if ".tfrecord" in user_var:
                preferences.dataset_test = user_input
            else:
                err_message("Testing Dataset must be a .tfrecord file")
                user_input = INPUT_ERR

        # dataset_train - File(string)
        elif user_var == defaults.DATASET_TRAIN_VAR != INPUT_ERR:
            if ".tfrecord" in user_var:
                preferences.dataset_train = user_input
            else:
                err_message("Training Dataset must be a .tfrecord file")
                user_input = INPUT_ERR

        # epochs - INT
        elif user_var == defaults.EPOCH_NUM_VAR and user_input != INPUT_ERR:
            preferences.epochs = user_input

        # image_size - INT(256 and 416)
        elif user_var == defaults.IMAGE_SIZE_VAR and user_input != INPUT_ERR:
            if  int(user_var) == 256 or int(user_var) == 416:
                preferences.image_size = user_input
            else:
                err_message("Image size must be 256 or 416")
                user_input = INPUT_ERR

        # max_checkpoints - INT
        elif user_var == defaults.MAX_CHECK_VAR:
            preferences.max_checkpoints = user_input

        # max_saved_sess - INt
        elif user_var == defaults.MAX_SESS_VAR:
            preferences.max_saved_sess = user_input

        # mode - List
        elif user_var == defaults.MODE_VAR:
            if check_input(user_input, defaults.MODE_OPTIONS) != INPUT_ERR:
                preferences.mode = user_input
            else:
                err_message(user_input + " not a valid mode option")
                user_input = INPUT_ERR

        # output - File(string)
        elif user_var == defaults.OUTPUT_VAR:
            preferences.output = user_input

        # sessions - INT
        elif user_var == defaults.SAVED_SESS_VAR:
            preferences.sessions = user_input

        # tiny - Boolean
        elif user_var == defaults.TINY_WEIGHTS_VAR:
            preferences.tiny = user_input

        # transfer - List
        elif user_var == defaults.TRANSFER_VAR:
            if check_input(user_input, defaults.TRANSFER_OPTIONS) != INPUT_ERR:
                preferences.transfer = user_input
            else:
                err_message(user_input + " not a valid transfer option")
                user_input = INPUT_ERR

        # validate_img_num - INT
        elif user_var == defaults.VALID_IMGS_VAR:
            preferences.validate_img_num = user_input

        # validate_input - File(String)
        elif user_var == defaults.VALID_IN_VAR:
            preferences.validate_input = user_input

        # weight_num_classes
        elif user_var == defaults.WEIGHTS_NUM_VAR:
            preferences.weight_num_classes = user_input

        # weights_path
        elif user_var == defaults.WEIGHTS_PATH_VAR:
            if ".tf" in user_var or ".weights" in user_var:
                preferences.dataset_train = user_input
            else:
                err_message("Weights must be a .tf or .weights file")
                user_input = INPUT_ERR

    if user_input != INPUT_ERR:
        print("\n\tSet " + str(user_var) + " to " + str(user_input) + "\n")

    return user_input

########################## LOAD #############################
# Description: takes in a preference file and changes the variables to the
#              values within the values
# Parameters: pref_path - string - user input, name of file with preferences
# Return: Nothing
def load(pref_path):

    # initialize variables
    failed = []         # list of all variables that failed
    changed = []        # list of all variables that were changed

    # adds .txt extention if needed
    if ".txt" not in pref_path:
        pref_path += ".txt"

    # adds the working directory if just the file name was given
    if not path.exists(pref_path):
        pref_path = os.getcwd() + "/" + pref_path

    # checks if file exists, if not returns error message
    if not path.exists(pref_path):
        err_message("Bad Preferences File, could not find file")
        return

    # Open file and set new preferences
    preferences.pref_file = pref_path
    with open(pref_path, "r") as f:

        # for each line in the preference file
        for line in f.readlines():
            value = file_utils.get_input_value(line, SPLIT_CHAR)    # value of the variable
            variable =  file_utils.get_input_var(line, SPLIT_CHAR)  # name of the variable
            modified = modify(variable, value)                      # boolean if the varible was last_modified

            # if variable was modified add to changed, else add to modifed
            if modified == NO_ERROR:
                changed.append(varibale + SPECIAL_CHAR + value)
            elif modifed == INPUT_ERROR:
                failed.append(varibale)

    # Print all variabels that were changed
    if len(changed) != 0:
        print("\n\tValues changed:")
        for item in changed:
            item = item.split(SPECIAL_CHAR)
            print("\t\t -" + item[0] + " to " + item[1])
        print("\t\tUse the display(d) command to current preferences")
    else:
        print("\n\tNo values were altered")

    # Print all variables that failed to change
    if len(failed) != 0:
        print("\n\n\tWARNING: The following items didn't change due incompatible input or incorrect file path:" )
        for item in failed:
            print("\t\t -" + item)
        print("\t\tUse the modify(m) command with no arguments to see accepted types")


########################## SAVE #############################
# Description: saves all current preference variables to a file
# Parameters: save_path - string - file name that prefereencs are saved
# Return: Nothing
def save(save_path):

    # initialize variabels
    changed_name = False    # if file has already been used
    files = 0               # nubers of files that have already been used

    if "." in save_path:
        index_before_suffix = save_path.rfind(".")
        save_path = save_path[0:index_before_suffix]

    if ".txt" not in save_path:
        new_file = save_path + ".txt"
    while os.path.exists(new_file):
        files += 1
        new_file = save_path + "-" + str(files) + ".txt"
        changed_name = True

    # if file given was already taken
    if changed_name:
        print("\n\tFile " + save_path + " is already a file, using " + new_file + " instead")

    # open preference file and write all variables to it
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

    print("\n\tNew preference path " + save_path + " successfully saved!")


########################## RUN #############################
# Description: runs the workbench
# Parameters: start_from - string - value to  determine if run or continue workbench
#             start_path - string - where the workbecnh is starting form
# Return: Nothing
def run(start_from, start_path):

    # start timer
    start_workbench_time = time.perf_counter()

    # check if necessary files exist
    # run was called, start from beginning
    # Setting for memory growth from old train_workbench.py
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if start_from == START:
        total_images = file_utils.check_files_exist(defaults.IMAGES_PATH,
                                                    defaults.DATA_PATH,
                                                    defaults.MIN_IMAGES,
                                                    preferences.output,
                                                    preferences.sessions,
                                                    defaults.TEST_IMAGE_PATH,
                                                    defaults.TRAIN_IMAGE_PATH,
                                                    defaults.VALIDATE_IMAGE_PATH,
                                                    preferences.weights)

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
        file_utils.save_session(defaults.OUTPUT_PATH, preferences.output, preferences.sessions,
                                preferences.max_saved_sess)
        print("\tDone!\n")

        # convert to checkpoint
        if preferences.transfer == "darknet":
            print("\nConverting darknet records to checkpoint...\n")
            convert_weights.run_weight_convert(preferences.weights,
                                               preferences.output + "/yolov3.tf",
                                               preferences.tiny,
                                               preferences.weight_num_classes)
            weights = (preferences.output + "/yolov3.tf").replace("\\", "/")
        else:
            weights = None

        print("\tCheckpoint Converted!\n")

    # if training
    if (start_from == CONTINUE and start_path != NONE) or start_from == START:

        # continue training from previous checkpoint
        if start_from != START:
            weights = start_path
            if os.path.isdir(weights):
                weights = file_utils.get_last_checkpoint(weights)
                weights = (weights.split(".tf")[0] + ".tf").replace("\\", "/")
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
            if preferences.transfer == "none":
                print("\n\tTraining from scratch...")
                transfer_mode = preferences.transfer
            else:
                print("\n\tTraining via " + preferences.transfer + "transfer")
                transfer_mode = preferences.transfer

        # start training
        start_train_time = time.perf_counter()
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
            print("\n\tTraining Failed!\n")
            return
        print("\n\tTraining Complete!\n")

    training_time = time.perf_counter() - start_train_time

    if (start_from == CONTINUE or start_from == START):
        if not file_utils.is_valid(preferences.output):
            err_message(preferences.output + " not found or is empty")
            return

        if not file_utils.is_valid(preferences.classifier_file):
            err_message(preferences.classifier_file + " not found or is empty")
            return

        # update checkpoint file
        chkpnt_weights = file_utils.get_last_checkpoint(preferences.output)
        chkpnt_weights = (chkpnt_weights.split(".tf")[0] + ".tf").replace("\\", "/")

        if chkpnt_weights == file_utils.ERROR or file_utils.CHECKPOINT_KEYWORD not in chkpnt_weights:
            err_message("No valid checkpoints found in " + file_utils.from_workbench(preferences.output))
            print("\t\tPlease use a trained checkpoint (e.g " + file_utils.CHECKPOINT_KEYWORD + "1.tf )")
            return

        file_utils.rename_checkpoints(preferences.output, preferences.max_checkpoints)
        file_utils.write_to_checkpoint(chkpnt_weights, (preferences.output + "/checkpoint").replace("\\", "/"))

        if chkpnt_weights == file_utils.ERROR:
            err_message("No checkpoints found in " + start_path)
            return

        print("\n\tUsing checkpoint " + chkpnt_weights )

        start_path = chkpnt_weights

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

        # Create Core ML Model
        try:
            print("\nCreating a CoreML model...")

            create_coreml.export_coreml(preferences.output, chkpnt_weights)

            print("\n\tCore ML model created!\n")

        except Exception as e:
            err_message("Failed to create CoreML model: " + str(e))

    # generating tensorflow models
    if (start_from == TEST_IMAGE):
        test_img = start_path

    else:
        test_img = preferences.validate_input

    if not os.path.isdir(test_img) and file_utils.is_valid(test_img):
        print("\n\tTest image location not found " + test_img)
        return

    print("\nTesting Images...")
    if path.isfile(test_img):
        out_img = preferences.output.replace("\\", "/") + test_img.split(".")[0] + "-output.jpg"
        detect_img.run_detect(preferences.classifier_file,
                              chkpnt_weights,
                              preferences.tiny,
                              preferences.image_size,
                              test_img,
                              out_img,
                              preferences.num_classes)
    else:
        test_img = (test_img + "/").replace("\\", "/")
        for file in os.listdir(test_img):
            if '.jpg' in file:
                out_img = preferences.output.replace("\\", "/") + file.split(".")[0] + "-output.jpg"
                print(out_img)
                detect_img.run_detect(preferences.classifier_file,
                                      chkpnt_weights,
                                      preferences.tiny,
                                      preferences.image_size,
                                      test_img + file,
                                      out_img,
                                      preferences.num_classes)

    # Save Runtimes
    total_runtime = file_utils.convert_to_time(time.perf_counter() - start_workbench_time)
    train_runtime = file_utils.convert_to_time(training_time)

    if (start_from != TEST_IMAGE):
        print("\n\n=============================== Workbench Successful! ===============================\n")
        print("\tTotal Training Runtime : " + train_runtime)
        print("\tTotal Workbench Runtime: " + total_runtime)
    print("\n\tAll models and images saved in " + preferences.output + "\n")
    print("=====================================================================================\n")




############################## MAIN ##########################
def main():
    warnings.simplefilter("ignore")
    check_admin()
    print("\nWelcome to the Digital Roll Workbench")
    print("\nEnter 'help' or 'h' for a list of commands:")
    while True:
        try:
            try:
                userInput = input("\n<WORKBENCH>: ")

            except EOFError:
                print("\n\n\n\n------ Current process stopped by user ------")
                print("\nEnter 'help' or 'h' for a list of commands:")
                running = True
            print("\nGiven Input: " + userInput)

            # HELP
            if get_input(userInput) == "help" or get_input(userInput) == "h":
                print_to_terminal.help()

           # RUN
            elif get_input(userInput) == "run" or get_input(userInput) == "r":
                run(START, NONE)

            # LITE
            elif get_input(userInput) == "lite" or get_input(userInput) == "l":
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

            # TEST
            elif userInput[0:5] == "test " or userInput[0:2] == "t ":
                if userInput[0:2] == "t ":
                    img_path = userInput[2:]
                else:
                    img_path = userInput[5:]
                img_path.strip("\n\r")
                run(TEST_IMAGE, img_path)

            # CONTINUE from last checkpoint
            elif get_input(userInput) == "continue" or get_input(userInput) == "c":
                run(CONTINUE, NONE)

            # CONTINUE from user given file
            elif userInput[0:5] == "continue " or userInput[0:2] == "c ":
                if userInput[0:2] == "c ":
                    prev_check = userInput[2:]
                else:
                    prev_check = userInput[5:]
                run(CONTINUE, prev_check)

            # DISPLAY
            elif get_input(userInput) == "display" or get_input(userInput) == "d":
                print(print_to_terminal.current_pref())

            # INFO
            elif get_input(userInput) == "info" or get_input(userInput) == "i":
                print_to_terminal.info()

            # LOAD
            elif userInput[0:5] == "load " or userInput[0:2] == "l ":
                if userInput[0:2] == "l ":
                    pref_path = userInput[2:]
                else:
                    pref_path = userInput[5:]
                pref_path.strip("\n")
                load(pref_path)

            # SAVE from current working directory
            elif get_input(userInput) == "save" or get_input(userInput) == "s":
                files = 1
                new_files = os.getcwd() + "/preferences-" + str(files) + ".txt"
                while os.path.exists(new_files):
                    files += 1
                    new_files = os.getcwd() + "/preferences-"  + str(files) + ".txt"
                save(new_files)

            # SAVE to user specified path
            elif userInput[0:5] == "save " or userInput[0:2] == "s ":
                if userInput[0:2] == "s ":
                    save_path = userInput[2:]
                else:
                    save_path = userInput[5:]
                save_path.strip("\n\r")
                save(save_path)

            # MODIFY DISPLAY VARIABLES
            elif get_input(userInput) == "modify" or get_input(userInput) == "m":
                print_to_terminal.modify_commands()

            # MODIFY user given variable
            elif userInput[0:7] == "modify " or userInput[0:2] == "m ":
                error = False
                userInputArr = userInput.split(" ")
                if len(userInputArr) >= 3:
                    modify(userInputArr[1], ' '.join(userInputArr[2:]))
                else:
                    print("Incorrect arguments, please provide a variable and a value (i.e. batch_size 3)")

            # QUIT
            elif get_input(userInput) == "quit" or get_input(userInput) == "q":
                print("\n\tExiting workbench...")
                exit()
            else:
                # end of cases, inform the user that their input was invalid
                print("\nCommand not recognized, try 'help' or 'h' for a list of options")

        except KeyboardInterrupt:
            print("\n\n\n\n------ Current process stopped by user ------")
            print("\nEnter 'help' or 'h' for a list of commands:")
            running = True
main()
