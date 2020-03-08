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
    print("\tClassifier file      ==> " + defaults.CLASSIFIERS_VAR + "      ACCEPTS: path to file")
    print("\tNumber of Classes    ==X Automatically updated when classifier is changed")
    print("\tDataset test         ==> " + defaults.DATASET_TEST_VAR + "     ACCEPTS: path to folder with images")
    print("\tDataset train        ==> " + defaults.DATASET_TRAIN_VAR + "    ACCEPTS: path to folder")
    print("\tEpochs               ==> " + defaults.EPOCH_NUM_VAR + "           ACCEPTS: int")
    print("\tImage Size           ==> " + defaults.IMAGE_SIZE_VAR + "       ACCEPTS: int")
    print("\tMax Checkpoints      ==> " + defaults.MAX_CHECK_VAR + "  ACCEPTS: int")
    print("\tMax Sessions         ==> " + defaults.MAX_SESS_VAR + "     ACCEPTS: int")
    print("\tMode                 ==> " + defaults.MODE_VAR + "             ACCEPTS: fit, eager_fit, eager_tf")
    print("\tOutput Model         ==> " + defaults.OUTPUT_VAR + "           ACCEPTS: path to save location")
    print("\tTiny Weights         ==> " + defaults.TINY_WEIGHTS_VAR + "     ACCEPTS: true/false")
    print("\tTransfer             ==> " + defaults.TRANSFER_VAR + "         ACCEPTS: none, darknet, no_output, frozen, fine_tune")
    print("\tValid Images Extract ==> " + defaults.VALID_IMGS_VAR + "      ACCEPTS: int")
    print("\tValidate Image Input ==> " + defaults.VALID_IN_VAR + "   ACCEPTS: int")
    print("\tWeights Path         ==> " + defaults.WEIGHTS_PATH_VAR + "          ACCEPTS: path to file")
    print("\tWeighted Classes     ==X Automatically updated when weights is changed")
    print("\tPreference File      ==X Automatically your most recently loaded preference file")

def help():
    print("\n COMMANDS")
    print("\n continue or c                  ==> Continue workbench if training was manually stopped")
    print("\n                                        You can also continue from previous checkpoint or session")
    print("\n                                        example: continue ./saved_session/session_1")
    print("\n display or d                   ==> Displays current settings")
    print("\n                                       example: change batch_size 3")
    print("\n help or h                      ==> Brings up this help display")
    print("\n info or i                      ==> Displays information on the workbench values")
    print("\n modify or m <variable> <value> ==> Modifys the setting variable to a new value")
    print("\n                                       For lists of values use the modify(m) command and use ? as your variable")
    print("\n load or l <path to pref.txt>   ==> Loads a given .txt file as the current preference text")
    print("\n quit or q                      ==> Exits the Workbench")
    print("\n run or r                       ==> Starts the process of training and validation")
    print("\n                                    + Saves the model at given output location")
    print("\n                                      and creates a Apple CoreML converted version")
    print("\n save or s <new .txt path>      ==> Saves the current settings to the path + name given")
    print("\n                                       example: save C:\\Users\\new_pref.txt")
    print("\n test or t <path to image>      ==> Tests a given image using the last checkpoint")
    print("\n tflite or i                    ==> Converts the current model in at the current output into a tflite model")

def info():
    print("\n\t\t\t////////////////////////////////")
    print("\t\t\t//     Workbench Info Menu    //")
    print("\t\t\t////////////////////////////////\n")

    print("The following information are values that can be changed for the work bench, it is currently optimized for training from scratch\n")
    print("Values:")
    print("\t--" + defaults.BATCH_SIZE_VAR + ": batch size of training...........................................(Default: " + split_path(defaults.DEFAULT_BATCH_SIZE) + ")")
    print("\t--" + defaults.CLASSIFIERS_VAR + ": name of the classifier names file...............................(Default: " + split_path(defaults.CLASSIFIER_FILE) + ")")
    print("\t--" + defaults.DATASET_TEST_VAR + ": path to test tf record file....................................(Default: " + split_path(defaults.TEST_TF_RECORD_PATH) + ")")
    print("\t--" + defaults.DATASET_TRAIN_VAR + ": path to train tf record file..................................(Default: " + split_path(defaults.TRAIN_TF_RECORD_PATH) + ")")
    print("\t--" + defaults.EPOCH_NUM_VAR + ": number of epochs used for training...................................(Default: " + split_path(defaults.DEFAULT_EPOCH_NUM) + ")")
    print("\t--" + defaults.IMAGE_SIZE_VAR + ": of your the images being trained on..............................(Default: " + split_path(defaults.DEFAULT_IMAGE_SIZE) + ")")
    print("\t--" + defaults.MAX_CHECK_VAR + ": amount of checkpoints saved at a given time.................(Default: " + split_path(defaults.DEFAULT_MAX_CHECK) + ")")
    print("\t--" + defaults.MAX_SESS_VAR + ": amount of sessions saved at a given time.......................(Default: " + split_path(defaults.DEFAULT_MAX_SESS) + ")")
    print("\t--" + defaults.MODE_VAR + ": mode of the training...................................................(Default: " + split_path(defaults.DEFAULT_MODE) + ")")
    print("\t\tOptions:")
    print("\t\t\t(1) fit: model.fit")
    print("\t\t\t(2) eager_fit: model.fit(run_eagerly=True")
    print("\t\t\t(3) eager_tf: custom GradientTape\n")
    print("\t--" + defaults.OUTPUT_VAR + ": location where tf and core ml model will be saved....................(Default: " + split_path(defaults.OUTPUT_PATH) + ")")
    print("\t--" + defaults.PREF_VAR + ": file that contains preferences, this cannot be ran with other flags....(Default: " + split_path(defaults.NO_PREF_PATH) + ")")
    print("\t--" + defaults.TINY_WEIGHTS_VAR + ": training with the tiny weights or not..........................(Default: " + split_path(defaults.DEFAULT_WEIGHT_TYPE) + ")")
    print("\t\tOptions:")
    print("\t\t\t(1) True: tiny_weights")
    print("\t\t\t(2) False: yolov3_weights\n")
    print("\t--" + defaults.TRANSFER_VAR + " type of transfer used for training..................................(Default: " + split_path(defaults.DFEAULT_TRANSFER_TYPE) + ")")
    print("\t\tOptions:")
    print("\t\t\t(1) none: Training from scratch")
    print("\t\t\t(2) darknet:  Transfer darknet")
    print("\t\t\t(3) no_output: Transfer all but output")
    print("\t\t\t(4) frozen: Transfer and freeze all")
    print("\t\t\t(5) fine_tune: Transfer all and freeze darknet only\n")
    print("\t--" + defaults.WEIGHTS_NUM_VAR + ": number of classes the weights file is trained on...........(Default: " + split_path(defaults.DEFAULT_WEIGHT_NUM) + ")")
    print("\t--" + defaults.WEIGHTS_PATH_VAR + ": path to the weights file..................(Default: " + split_path(defaults.DEFAULT_WEIGHT_PATH) + ")")
    print("\t--" + defaults.VALID_IMGS_VAR + ": number of files removed from training data to test on.....(Default: " + split_path(defaults.DEFAULT_VALID_IMGS) + ")")
    print("\t--" + defaults.VALID_IN_VAR + ": path to image(s) you want to test the new model on.....(Default: " + split_path(defaults.INPUT_IMAGE_PATH) + ")")


def current_pref():
    print("\nCurrent Preferences:\n")
    print("\tBatch Size............. " + str(preferences.batch_size))
    print("\tClassifier file........ " + preferences.classifier_file)
    print("\tNumber of Classes...... " + str(preferences.num_classes))
    print("\tDataset test........... " + preferences.dataset_test)
    print("\tDataset train.......... " + preferences.dataset_train)
    print("\tEpochs................. " + str(preferences.epochs))
    print("\tImage Size............. " + str(preferences.image_size))
    print("\tMax Checkpoints........ " + str(preferences.max_checkpoints))
    print("\tMax Saved Sessions..... " + str(preferences.max_saved_sess))
    print("\tMode................... " + preferences.mode)
    print("\tOutput Model........... " + preferences.output)
    print("\tPreference File........ " + preferences.pref_file)
    print("\tTiny Weights........... " + str(preferences.tiny))
    print("\tTransfer............... " + preferences.transfer)
    print("\tValidate Image Num..... " + str(preferences.validate_img_num))
    print("\tValidate Image Input... " + preferences.validate_input)
    print("\tWeighted Classes....... " + str(preferences.weight_num_classes))
    print("\tWeights Path........... " + preferences.weights)
