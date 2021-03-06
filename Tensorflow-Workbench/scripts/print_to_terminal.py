import os
from scripts import defaults
from scripts import preferences

###### Split path
# Takes a path and returns everying after the workbench directory
def split_path(path):
    keyword = "Workbench"
    if isinstance(path, int) or isinstance(path, float):
        return str(path)
    try:
        if keyword in path:
            return "." + path.split(keyword)[1]
        else:
            return path
    except:
        return path


def modify_commands():
    print("\tBatch Size           ==> " + defaults.BATCH_SIZE_VAR + "       ACCEPTS: int")
    print("\tClassifier file      ==> " + defaults.CLASSIFIERS_VAR + "      ACCEPTS: path to a '.names' file")
    print("\tNumber of Classes    ==X Automatically updated when classifier is changed")
    print("\tDataset test         ==> " + defaults.DATASET_TEST_VAR + "     ACCEPTS: path to folder with images")
    print("\tDataset train        ==> " + defaults.DATASET_TRAIN_VAR + "    ACCEPTS: path to folder")
    print("\tEpochs               ==> " + defaults.EPOCH_NUM_VAR + "           ACCEPTS: int")
    print("\tImage Size           ==> " + defaults.IMAGE_SIZE_VAR + "       ACCEPTS: int only 256 and 416")
    print("\tMax Checkpoints      ==> " + defaults.MAX_CHECK_VAR + "  ACCEPTS: int")
    print("\tMax Sessions         ==> " + defaults.MAX_SESS_VAR + "     ACCEPTS: int")
    print("\tMode                 ==> " + defaults.MODE_VAR + "             ACCEPTS: fit, eager_fit, eager_tf")
    print("\tOutput               ==> " + defaults.OUTPUT_VAR + "           ACCEPTS: path to current session location")
    print("\tSessions             ==> " + defaults.SAVED_SESS_VAR + "         ACCEPTS: path to save previous sessions to")
    print("\tTiny Weights         ==> " + defaults.TINY_WEIGHTS_VAR + "     ACCEPTS: true/false")
    print("\tTransfer             ==> " + defaults.TRANSFER_VAR + "         ACCEPTS: none, darknet, no_output, frozen, fine_tune")
    print("\tValid Images Extract ==> " + defaults.VALID_IMGS_VAR + "      ACCEPTS: int")
    print("\tValidate Image Input ==> " + defaults.VALID_IN_VAR + "   ACCEPTS: int")
    print("\tWeights Path         ==> " + defaults.WEIGHTS_PATH_VAR + "          ACCEPTS: path to  a '.tf' or '.weights' file")
    print("\tWeighted Classes     ==X Automatically updated when weights is changed")
    print("\tPreference File      ==X Automatically your most recently loaded preference file")
    print("\tStart Path           ==> Changes the starting path for using the 'continue' command")

def help():
    print("\n COMMANDS")
    print("\n continue or c                  ==> Continue workbench from a checkpoint or session")
    print("\n                                        example continue   (this uses the most recent check in your output folder)")
    print("\n                                        example: continue ./output/yolov3_train_5.tf")
    print("\n                                        example: continue ./saved_session/session_1")
    print("\n display or d                   ==> Displays current settings")
    print("\n finish or f                    ==> Finishes the workbench if training was stopped manually")
    print("\n graph or g                     ==> Displays a graph showing the train and test loss of the last training session")
    print("\n help or h                      ==> Brings up this help display")
    print("\n info or i                      ==> Displays information on the workbench values")
    print("\n modify or m <variable> <value> ==> Modifys the setting variable to a new value")
    print("\n                                        For lists of values use the modify(m) command without arguments")
    print("\n load or l <path to pref.txt>   ==> Loads a given .txt file as the current preference text")
    print("\n quit or q                      ==> Exits the Workbench")
    print("\n run or r                       ==> Starts the process of training and validation")
    print("\n                                    + Saves the model at given output location")
    print("\n                                      and creates a Apple CoreML converted version")
    print("\n save or s <new .txt path>      ==> Saves the current settings to the path + name given")
    print("\n                                        example: save new_pref.txt")
    print("\n                                        if no argument given, saves in current working directory as preferences_<number>")
    print("\n test or t <path to image>      ==> Tests a given image using the last checkpoint")
    print("\n tflite or tf                    ==> Converts the current model in at the current output into a tflite model")

def info():
    print("\n\t\t\t////////////////////////////////")
    print("\t\t\t//     Workbench Info Menu    //")
    print("\t\t\t////////////////////////////////\n")

    print("The following information are values that can be changed for the work bench, it is currently optimized for training from scratch\n")
    print("Values:")
    print("\t--" + defaults.BATCH_SIZE_VAR + ": images trained at once...........................................(Default: " + split_path(defaults.DEFAULT_BATCH_SIZE) + ")")
    print("\t--" + defaults.CLASSIFIERS_VAR + ": name of the classifier names file...............................(Default: " + split_path(defaults.CLASSIFIER_FILE) + ")")
    print("\t--" + defaults.DATASET_TEST_VAR + ": path to test tf record file....................................(Default: " + split_path(defaults.TEST_TF_RECORD_PATH) + ")")
    print("\t--" + defaults.DATASET_TRAIN_VAR + ": path to train tf record file..................................(Default: " + split_path(defaults.TRAIN_TF_RECORD_PATH) + ")")
    print("\t--" + defaults.EPOCH_NUM_VAR + ": number of iterations thorugh the dataset.............................(Default: " + split_path(defaults.DEFAULT_EPOCH_NUM) + ")")
    print("\t--" + defaults.IMAGE_SIZE_VAR + ": amount of bounding boxes created per image.......................(Default: " + split_path(defaults.DEFAULT_IMAGE_SIZE) + ")")
    print("\t--" + defaults.MAX_CHECK_VAR + ": amount of checkpoints saved at a given time.................(Default: " + split_path(defaults.DEFAULT_MAX_CHECK) + ")")
    print("\t--" + defaults.MAX_SESS_VAR + ": amount of sessions saved at a given time.......................(Default: " + split_path(defaults.DEFAULT_MAX_SESS) + ")")
    print("\t--" + defaults.MODE_VAR + ": mode of the training...................................................(Default: " + split_path(defaults.DEFAULT_MODE) + ")")
    print("\t\tOptions:")
    print("\t\t\t(1) fit: model.fit")
    print("\t\t\t(2) eager_fit: model.fit(run_eagerly=True")
    print("\t\t\t(3) eager_tf: custom GradientTape\n")
    print("\t--" + defaults.OUTPUT_VAR + ": location where tf and core ml model will be saved....................(Default: " + split_path(defaults.OUTPUT_PATH) + ")")
    print("\t--" + defaults.PREF_VAR + ": file that contains preferences, this cannot be ran with other flags....(Default: " + split_path(defaults.NO_PREF_PATH) + ")")
    print("\t--" + defaults.SAVED_SESS_VAR + ": location that previous sessions are saved..........................(Default: " + split_path(defaults.SAVED_SESS_PATH) + ")")
    print("\t--" + defaults.TINY_WEIGHTS_VAR + ": training with the tiny weights or not..........................(Default: " + split_path(defaults.DEFAULT_WEIGHT_TYPE) + ")")
    print("\t\tOptions:")
    print("\t\t\t(1) True: tiny_weights")
    print("\t\t\t(2) False: not tiny_weights\n")
    print("\t--" + defaults.TRANSFER_VAR + " type of transfer used for training..................................(Default: " + split_path(defaults.DFEAULT_TRANSFER_TYPE) + ")")
    print("\t\tOptions:")
    print("\t\t\t(1) none: Training from scratch")
    print("\t\t\t(2) darknet:  Transfer darknet")
    print("\t\t\t(3) no_output: Transfer all but output")
    print("\t\t\t(4) frozen: Transfer and freeze all")
    print("\t\t\t(5) fine_tune: Transfer all and freeze darknet only\n")
    print("\t--" + defaults.WEIGHTS_NUM_VAR + ": number of classes the weights file is trained on...........(Default: " + split_path(defaults.DEFAULT_WEIGHT_NUM) + ")")
    print("\t--" + defaults.WEIGHTS_PATH_VAR + ": path to the weights file............................................(Default: " + split_path(defaults.DEFAULT_WEIGHT_PATH) + ")")
    print("\t--" + defaults.VALID_IMGS_VAR + ": number of files removed from training data to test on...........(Default: " + split_path(defaults.DEFAULT_VALID_IMGS) + ")")
    print("\t--" + defaults.VALID_IN_VAR + ": path to image(s) you want to test the new model on...........(Default: " + split_path(defaults.INPUT_IMAGE_PATH) + ")")


def current_pref():
    string = ""
    string += "\tBatch Size............. " + str(preferences.batch_size) + "\n"
    string += "\tClassifier file........ " + str(preferences.classifier_file) + "\n"
    string += "\tNumber of Classes...... " + str(preferences.num_classes) + "\n"
    string += "\tDataset test........... " + str(preferences.dataset_test) + "\n"
    string += "\tDataset train.......... " + str(preferences.dataset_train) + "\n"
    string += "\tEpochs................. " + str(preferences.epochs) + "\n"
    string += "\tImage Size............. " + str(preferences.image_size) + "\n"
    string += "\tMax Checkpoints........ " + str(preferences.max_checkpoints) + "\n"
    string += "\tMax Saved Sessions..... " + str(preferences.max_saved_sess) + "\n"
    string += "\tMode................... " + str(preferences.mode) + "\n"
    string += "\tOutput Model........... " + str(preferences.output) + "\n"
    string += "\tSave Sessions.......... " + str(preferences.sessions) + "\n"
    string += "\tPreference File........ " + str(preferences.pref_file) + "\n"
    string += "\tTiny Weights........... " + str(preferences.tiny) + "\n"
    string += "\tTransfer............... " + str(preferences.transfer) + "\n"
    string += "\tValidate Image Num..... " + str(preferences.validate_img_num) + "\n"
    string += "\tValidate Image Input... " + str(preferences.validate_input) + "\n"
    string += "\tWeighted Classes....... " + str(preferences.weight_num_classes) + "\n"
    string += "\tWeights Path........... " + str(preferences.weights) + "\n"
    return string

def print_both(string, file):
    with open(file, 'a') as f:
        f.write(string)
    print(string)
