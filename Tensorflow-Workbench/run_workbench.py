# import packages
import os
import sys
from os import path
from os.path import dirname
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
    from matplotlib import pyplot
    from tensorflow.keras.applications import MobileNet

    from yolov3_tf2.models import (
        YoloV3, YoloV3Tiny
    )

    test_checkpoint = file_utils.get_last_checkpoint(preferences.output)
    output_file = file_utils.get_output_file("./logs/")
    SPLIT_CHAR = "="
    NONE = ""
    RUN = 1001
    CONTINUE = 1002
    SINGLE = 1003
    TEST_IMAGE = 1004
    SPECIAL_CHAR = "?<#>@"
    INPUT_ERR = -99999999
    NO_ERROR = -99999998
    training_data = []
    testing_data = []

# if script not found
except FileNotFoundError as e:
    print("\n\n\tERROR: " + str(e))
    print("\n\t\tEnsure that:")
    print("\t\t        - The scripts folder and yolov3_tf2 folder have not been removed or altered")
    print("\t\t        - You are in the proper directory")
    print("\n\t\tAfter ensuring necessary files are in your directory and re-run the workbench\n\n")
    exit()

# if package not found
except ImportError as e:
    print("\n\n\tERROR: " + str(e))
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
    return input.lstrip().lower()


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
            if os.path.isdir(value):
                value = (value + "/").replace("//", "/")
            return value
        elif value.lower().replace(" ", "") != "none":
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
        value = value.replace(" ", "")
        value = value.replace("\n", "")
        if value in type:
            return value

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

    # check if it is variable that should be a File
    elif user_var == defaults.CLASSIFIERS_VAR or \
         user_var == defaults.DATASET_TEST_VAR or \
         user_var == defaults.DATASET_TRAIN_VAR or \
         user_var == defaults.OUTPUT_VAR or \
         user_var == defaults.SAVED_SESS_VAR or \
         user_var == defaults.VALID_IN_VAR or \
         user_var == defaults.WEIGHTS_PATH_VAR:

         temp = check_input(user_input, defaults.FILE)
         if temp != INPUT_ERR:
             try:
                user_input = temp.lower()
             except:
                 user_input = INPUT_ERR
         elif user_var == defaults.WEIGHTS_PATH_VAR and user_input.lower().replace(" ", "") == "none":
            user_input = None

         else:
             user_input = INPUT_ERR

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
            if ".tfrecord" in user_input:
                preferences.dataset_test = user_input
            else:
                err_message("Testing Dataset must be a .tfrecord file")
                user_input = INPUT_ERR

        # dataset_train - File(string)
        elif user_var == defaults.DATASET_TRAIN_VAR != INPUT_ERR:
            if ".tfrecord" in user_input:
                preferences.dataset_train = user_input
            else:
                err_message("Training Dataset must be a .tfrecord file")
                user_input = INPUT_ERR

        # epochs - INT
        elif user_var == defaults.EPOCH_NUM_VAR and user_input != INPUT_ERR:
            preferences.epochs = user_input

        # image_size - INT(256 and 416 and 224)
        elif user_var == defaults.IMAGE_SIZE_VAR and user_input != INPUT_ERR:
            if user_input == 256 or user_input == 416 or user_input == 224:
                preferences.image_size = user_input
            else:
                err_message("Image size must be 224, 256 or 416")
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
            user_input = (user_input + "/").replace("//", "/")
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
            if user_input == None or ".tf" in user_input or ".weights" in user_input:
                preferences.weights = user_input
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

    print("\n=======================================================\n Loading Prferences...\n\n")
    # Open file and set new preferences
    preferences.pref_file = pref_path
    with open(pref_path, "r") as f:

        # for each line in the preference file
        for line in f.readlines():
            value = file_utils.get_input_value(line, SPLIT_CHAR)    # value of the variable
            variable =  file_utils.get_input_var(line, SPLIT_CHAR)  # name of the variable
            print("Found " + variable)
            modified = modify(variable, value)                      # boolean if the varible was last_modified

            # if variable was modified add to changed, else add to modifed
            if modified == INPUT_ERR:
                failed.append(variable)
            else:
                changed.append(variable + SPECIAL_CHAR + value)

    print("=======================================================\n")
    # Print all variabels that were changed
    if len(changed) != 0:
        print("\n\tValues changed:")
        for item in changed:
            item = item.split(SPECIAL_CHAR)
            print("\t\t -" + item[0] + " to " + item[1])
        print("\n\t\tUse the display(d) command to current preferences")
    else:
        print("\n\tNo values were altered")

    # Print all variables that failed to change
    if len(failed) != 0:
        print("\n\n\tWARNING: The following items didn't change due incompatible input or incorrect file path:" )
        for item in failed:
            print("\t\t -" + item)
        print("\n\t\tUse the modify(m) command with no arguments to see accepted types")


########################## SAVE #############################
# Description: saves all current preference variables to a file
# Parameters: save_path - string - file name that prefereencs are saved
# Return: Nothing
def save(save_path):

    # initialize variabels
    changed_name = False      # if file has already been used
    files = 0                 # nubers of files that have already been used
    path = os.getcwd()
    file_name = "preferences"
    file_changed = False
    BACKSLASH = "\ ".replace(" ", "")

    # if user included filename
    if ".txt" in save_path:
        temp_path = dirname(save_path)
        file_name  = save_path.replace(temp_path + BACKSLASH, "").replace(".txt", "")
        save_path = temp_path
        file_changed = True

    # if user give filepath exists
    if os.path.exists(save_path):
        path = save_path

    # else using current working directory if not already
    elif save_path or save_path == ".":
        err_message("Path " + save_path + " does not exist")
        print("\t\tUsing current directory")
        if file_changed:
            print("\t\tKeeping filename " + file_name + ".txt")

    new_file = path + BACKSLASH + file_name + ".txt"

    # if file aready exists append with a dash and number
    while os.path.exists(new_file):
        files += 1
        new_file = path + BACKSLASH + file_name + "-"  + str(files) + ".txt"

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

    print("\n\tNew preference file " + new_file + " successfully saved!")


########################## RUN #############################
# Description: runs the workbench
# Parameters: start_from - string - value to  determine if run or continue workbench
#             start_path - string - where the workbench is starting form
# Return: Nothing
def run(start_from, start_path):

    # initialize variables
    start_workbench_time = time.perf_counter()   # Start Workbench timer
    training_time = None                         # Set Training timer variable

    # Check if GPU is present
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("\n\tWARNING: You are not currently using GPU, it is not recommended to train a model using just cpu\n")

    # If starting from scratch
    if start_from == RUN:
        total_images = file_utils.check_files_exist(defaults.IMAGES_PATH,
                                                    defaults.DATA_PATH,
                                                    defaults.MIN_IMAGES,
                                                    preferences.output,
                                                    preferences.sessions,
                                                    defaults.TEST_IMAGE_PATH,
                                                    defaults.TRAIN_IMAGE_PATH,
                                                    defaults.VALIDATE_IMAGE_PATH,
                                                    preferences.weights,
                                                    preferences.transfer)

        # check if any images found, if not print error message and exit run command
        if total_images == 0:
            err_message("No images have been found in the image folder")
            print("\t\tImage Folder Location: " + defaults.IMAGES_PATH)
            print(
                "\n\t\tFor an example set, look at the Pre_Labeled_Images folder in the repository or at https://github.com/tylerboice/Digital-Roll\n")
            return

        # check if images found meets the required amount, if not print error messafe and exit run command
        elif total_images < defaults.MIN_IMAGES:
            err_message("Workbench needs at minimum " + str(defaults.MIN_IMAGES) + " to train. Current Count: " + str(total_images))
            print("\t       However it is recommended you have around 1000 per classifier")
            return

        # print total images found
        print("\nTraining Data Info:")
        print("\n\tTotal images = " + str(total_images))

        # counts all objects labeled and saves them in all_classifers
        all_classifiers = file_utils.get_classifiers(defaults.IMAGES_PATH)
        # counts all unique objects labeled
        classifiers = file_utils.classifier_remove_dup(all_classifiers)
        preferences.num_classes = len(classifiers)
        # resets classifier list to save memory
        all_classifiers = []

        # creates classifier file using the unique classifiers found if any found, if not return
        if len(classifiers) == 0:
            err_message("No Classifiers found, make sure you labelled the images")
            return
        file_utils.create_classifier_file(preferences.classifier_file, classifiers)


        # sort all the images into TRAIN, TEST, AND VALIDATE folders
        print("\nSorting images...")
        file_utils.sort_images(preferences.validate_img_num,
                               defaults.IMAGES_PATH,
                               defaults.TEST_IMAGE_PATH,
                               defaults.TRAIN_IMAGE_PATH,
                               preferences.validate_input,
                               total_images)
        print("\n\tAll images sorted!\n\n")

        # generate train.tfrecord and test.tfrecord
        print("Generating images and xml files into tfrecords...\n")
        generate_tf.generate_tfrecords(defaults.TRAIN_IMAGE_PATH,
                                       preferences.dataset_train)
        generate_tf.generate_tfrecords(defaults.TEST_IMAGE_PATH,
                                       preferences.dataset_test)
        print("\n\tSuccessfully generated tf records!\n")

        # save previous session
        print("\nChecking for previous Sessions...\n")
        file_utils.save_session(defaults.OUTPUT_PATH,
                                preferences.output,
                                preferences.sessions,
                                preferences.max_saved_sess)
        print("\tDone!\n")

        # convert to checkpoint if nessacary, if not train from scratch
        if preferences.transfer == "darknet":
            try:
                print("\nConverting darknet records to checkpoint...\n")
                convert_weights.run_weight_convert(preferences.weights,
                                                   preferences.output + "/yolov3.tf",
                                                   preferences.tiny,
                                                   preferences.weight_num_classes)
                weights = (preferences.output + "/yolov3.tf").replace("\\", "/").replace("//", "/")
                print("\tCheckpoint Converted!\n")
                
            except Exception as e:
                err_message("Conversion failed: " + str(e))
        else:
            weights = None

    # if continue or run command was called
    if (start_from == CONTINUE and start_path != NONE) or start_from == RUN:

        # If continue training from previous checkpoint
        if start_from == CONTINUE:
            weights = start_path

            # if folder given, search folder for most recent checkpoint
            if os.path.isdir(weights):
                weights = file_utils.get_last_checkpoint(weights)
                weights = (weights.split(".tf")[0] + ".tf").replace("\\", "/").replace("//", "/")

            # check to make sure file is correct type, if not return
            if ".tf" not in weights:
                err_message("File is not a checkpoint")
                print("\n\t\tCheckpoint Example: yolov3_train_3.tf")
                return

            # if checkpoint not found and using preferences.output, return error
            if "ERROR" in weights and "ERROR" in start_path:
                err_message("No checkpoints found in " + start_path)
                return

            print("\n\tContinuing from " + weights)
            print("\nResume Training...")
            transfer_mode = 'fine_tune'

        # run command was called, train from scratch
        else:
            print("\nBegin Training...")
            if preferences.transfer == "none":
                print("\n\tTraining from scratch, this will take some time...\n")
            else:
                print("\n\tTraining via " + preferences.transfer + " transfer, this will take some time...")
            transfer_mode = preferences.transfer

        # TRAINING
        type = 'yolo'
        if(type == 'resnet'):
            try:
                # start training counter
                start_train_time = time.perf_counter()
                # start training
                train, test = train_workbench.train_resnet(preferences.dataset_train,
                                                        preferences.dataset_test,
                                                        defaults.IMAGES_PATH,
                                                        preferences.classifier_file,
                                                        preferences.mode,
                                                        preferences.image_size,
                                                        preferences.epochs,
                                                        preferences.batch_size,
                                                        defaults.DEFAULT_LEARN_RATE,
                                                        preferences.num_classes,
                                                        preferences.output)
                training_data.extend(train)
                testing_data.extend(test)
                training_time = time.perf_counter() - start_train_time
                print("\n\n\tTraining Complete!\n")

            # Training Failed
            except Exception as e:
                err_message("Training Failed: " + str(e))
                return
        # YOLO Training method call
        else:
            try:
                # start training counter
                start_train_time = time.perf_counter()
                # start training
                train, test = train_workbench.run_train(preferences.dataset_train,
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
                training_data.extend(train)
                testing_data.extend(test)
                training_time = time.perf_counter() - start_train_time
                print("\n\n\tTraining Complete!\n")

            # Training Failed
            except Exception as e:
                err_message("Training Failed: " + str(e))
                return


    # Create Models and test validate images
    if (start_from == CONTINUE or start_from == RUN):

        # Ensure output exists and is not empty
        if not file_utils.is_valid(preferences.output):
            err_message(preferences.output + " not found or is empty")
            return

        # Ensure classifier.names file exists and is not empty
        if not file_utils.is_valid(preferences.classifier_file):
            err_message(preferences.classifier_file + " not found or is empty")
            return

        try:
            file_utils.rename_checkpoints(preferences.output)
        except Exception as e:
            err_message(e)

        # update checkpoint file
        chkpnt_weights = file_utils.get_last_checkpoint(preferences.output)
        chkpnt_weights = (chkpnt_weights.split(".tf")[0] + ".tf").replace("//", "/")
        # Rename checkpoints to ensure the newest one has the greatest number

        file_utils.write_to_checkpoint(chkpnt_weights, (preferences.output + "/checkpoint").replace("\\", "/").replace("//", "/"))
        # If Error happend in getting new checkpoint
        if chkpnt_weights == file_utils.ERROR or file_utils.CHECKPOINT_KEYWORD not in chkpnt_weights:
            err_message("No valid checkpoints found in " + file_utils.from_workbench(preferences.output))
            print("\t\tPlease use a trained checkpoint (e.g " + file_utils.CHECKPOINT_KEYWORD + "1.tf )")
            return
        print(preferences.output)
        # if no checkpoint found, return
        if chkpnt_weights == file_utils.ERROR:
            err_message("No checkpoints found in " + start_path)
            return

        start_path = chkpnt_weights
        test_img = preferences.validate_input.lower()

        # GENERATING TENSORFLOW MODEL
        print("\nGenerating TensorFlow model...\n")
        if os.path.isdir(test_img):
            for file in os.listdir(preferences.validate_input):
                file = file.lower()
                if file.endswith(tuple(file_utils.IMAGE_TYPES)) :
                    test_img = preferences.validate_input + file
        if not test_img.endswith(tuple(file_utils.IMAGE_TYPES)):
            print("No validate images found, ensure you have a valid image type in the ./images/validate folder")
            return

        try:  # create model
            create_tf_model.run_export_tfserving(chkpnt_weights,
                                                 preferences.tiny,
                                                 preferences.output,
                                                 preferences.classifier_file,
                                                 test_img,
                                                 preferences.num_classes)

            print("\n\t Successfully created TensorFlow model\n")

        except Exception as e: # if model fails, return
            err_message("Failed to create TensorFlow model: " + str(e))
            return

        # Create Core ML Model
        try:
            print("\nCreating a CoreML model...\n")
            create_coreml.export_coreml(preferences.output, chkpnt_weights)
            print("\n\tCore ML model created!\n")

        except Exception as e: # if CoreML model fails print error message
            err_message("Failed to create CoreML model: " + str(e))

    # generating tensorflow models
    if (start_from == TEST_IMAGE):
        test_img = start_path

        chkpnt_weights = file_utils.get_last_checkpoint(preferences.output)
        chkpnt_weights = (chkpnt_weights.split(".tf")[0] + ".tf").replace("//", "/")

        # Rename checkpoints to ensure the newest one has the greatest number
        try:
            file_utils.rename_checkpoints(preferences.output)
            file_utils.write_to_checkpoint(chkpnt_weights, (preferences.output + "/checkpoint").replace("\\", "/").replace("//", "/"))
        except Exception as e:
            err_message("Could not rename files due to: " + str(e))

        # Get checkpoint
        chkpnt_weights = file_utils.get_last_checkpoint(preferences.output)
        chkpnt_weights = (chkpnt_weights.split(".tf")[0] + ".tf").replace("\\", "/").replace("//", "/")

        # If Error happend in getting new checkpoint
        if chkpnt_weights == file_utils.ERROR or file_utils.CHECKPOINT_KEYWORD not in chkpnt_weights:
            err_message("No valid checkpoints found in " + file_utils.from_workbench(preferences.output))
            print("\t\tPlease use a trained checkpoint (e.g " + file_utils.CHECKPOINT_KEYWORD + "1.tf )")
            return

        # if no checkpoint found, return
        if chkpnt_weights == file_utils.ERROR:
            err_message("No checkpoints found in " + start_path)
            return

    else:
        test_img = preferences.validate_input

    if not os.path.isdir(test_img) and file_utils.is_valid(test_img):
        print("\n\tTest image location not found " + test_img)
        return

    # TEST VALIDATE IMAGES AGAINST MODEL
    print("\nTesting Image(s)...")
    files_found = False
    # test_img given was a signle file, test file
    if path.isfile(test_img):
        test_img = test_img.lower()
        if test_img.endswith(tuple(file_utils.IMAGE_TYPES)) or test_img.endswith(file_utils.XML_TYPE):
            if test_img.endswith(file_utils.XML_TYPE):
                extract_img_from_xml(test_img)
                test_img = test_img.split(".")[0] + ".jpg"
            files_found = True
            file_type = file_utils.get_type(test_img)
            out_img = preferences.output + test_img.split(".")[0] + "-output" + file_type
            detect_img.run_detect(preferences.classifier_file,
                                  chkpnt_weights,
                                  preferences.tiny,
                                  preferences.image_size,
                                  test_img,
                                  out_img,
                                  preferences.num_classes)

    # test_img given was a signle folder, test entire folder
    elif path.isdir(test_img):
        test_img = (test_img + "/").replace("\\", "/").replace("//", "/")
        for file in os.listdir(test_img):
            file = file.lower()
            if file.endswith(tuple(file_utils.IMAGE_TYPES)):
                files_found = True
                file_type = file_utils.get_type(file)
                out_img = preferences.output + file.split(".")[0] + "-output" + file_type
                detect_img.run_detect(preferences.classifier_file,
                                      chkpnt_weights,
                                      preferences.tiny,
                                      preferences.image_size,
                                      test_img + file,
                                      out_img,
                                      preferences.num_classes)
    else:
        files_found = True
        err_message("Test file(s) not found " + file_utils.from_workbench(test_img))

    if not files_found:
        err_message( file_utils.from_workbench(test_img) + " is not an image or does not contain images")


    # Save Runtimes
    total_runtime = file_utils.convert_to_time(time.perf_counter() - start_workbench_time)
    if training_time != None:
        train_runtime = file_utils.convert_to_time(training_time)

    # Finsh workbench output
    if (start_from != TEST_IMAGE):
        print("\n\n=============================== Workbench Successful! ===============================\n")
        if training_time != None:
            print("\tTotal Training Runtime : " + train_runtime)
        print("\tTotal Workbench Runtime: " + total_runtime)
        print("\n\tAll models and images saved in " + file_utils.from_workbench(preferences.output) + "\n")
        print("=====================================================================================\n")


def plot_test_train(train_data, test_data):
    pyplot.plot(train_data, label='train')
    pyplot.plot(test_data, label='test')
    # Use a fork for the display to allow the main thread to finish so that the graph doesn't halt the system
    pyplot.legend()
    pyplot.show()


############################## MAIN ##########################
def main():
    tf.keras.backend.clear_session()
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
            print("\nGiven Input: " + userInput)

            userInput = get_input(userInput)

            # CONTINUE from last checkpoint
            if userInput == "continue" or userInput == "c":
                try:
                    last_weight = file_utils.get_last_checkpoint(preferences.output)
                    last_weight = (last_weight.split(".tf")[0] + ".tf").replace("\\", "/").replace("//", "/")
                    run(CONTINUE, last_weight)
                except Exception as e:
                    err_message(str(e))

            # CONTINUE from user given file
            elif userInput[0:8] == "continue " or userInput[0:2] == "c ":
                if userInput[0:2] == "c ":
                    prev_check = userInput[2:]
                else:
                    prev_check = userInput[5:]
                try:
                    run(CONTINUE, prev_check)
                except Exception as e:
                    err_message(str(e))

            elif userInput == "finish" or userInput == "f":
                try:
                    run(CONTINUE, NONE)
                except Exception as e:
                    err_message(str(e))

            elif userInput == "graph" or userInput == "g":
                plot_test_train(training_data, testing_data)

            # DISPLAY
            elif userInput == "display" or userInput == "d":
                print(print_to_terminal.current_pref())


            # HELP
            elif userInput == "help" or userInput == "h":
                print_to_terminal.help()

            # INFO
            elif userInput == "info" or userInput == "i":
                print_to_terminal.info()

            # LITE
            elif userInput == "lite" or userInput == "tf":
                # convert model to tensorflow lite for android use
                print("WARNING: This method is still untested and in development.")
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

            # LOAD
            elif userInput == "load" or userInput == "l":
                err_message("Load requires an argument of the preference file")

            elif userInput[0:5] == "load " or userInput[0:2] == "l ":
                if userInput[0:2] == "l ":
                    pref_path = userInput[2:]
                else:
                    pref_path = userInput[5:]
                pref_path.strip("\n")
                load(pref_path)

            # MODIFY DISPLAY VARIABLES
            elif userInput == "modify" or userInput == "m":
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
            elif userInput == "quit" or userInput == "q":
                print("\n\tExiting workbench...")
                tf.keras.backend.clear_session()
                exit()

            # RUN
            elif userInput == "run" or userInput == "r":
                try:
                    run(RUN, NONE)
                except Exception as e:
                    err_message(str(e))


            # SAVE from current working directory
            elif userInput == "save" or userInput == "s":
                save(os.getcwd())

            # SAVE to user specified path
            elif userInput[0:5] == "save " or userInput[0:2] == "s ":
                if userInput[0:2] == "s ":
                    save_path = userInput[2:]
                else:
                    save_path = userInput[5:]
                try:
                    save(save_path.strip("\n\r"))
                except Exception as e:
                    err_message(str(e))

            # TEST
            elif userInput == "test" or userInput == "t":
                err_message("Test requires an argument of the image or image folder you want to test")

            elif userInput[0:5] == "test " or userInput[0:2] == "t ":
                if userInput[0:2] == "t ":
                    img_path = userInput[2:]
                else:
                    img_path = userInput[5:]

                try:
                    run(TEST_IMAGE, img_path.strip("\n\r"))
                except Exception as e:
                    err_message(str(e))

            else:
                # end of cases, inform the user that their input was invalid
                print("\n\tCommand not recognized, try 'help' or 'h' for a list of options")

        except KeyboardInterrupt:
            print("\n\n\n\n------ Current process stopped by user ------")
            print("\nEnter 'help' or 'h' for a list of commands:")
main()
