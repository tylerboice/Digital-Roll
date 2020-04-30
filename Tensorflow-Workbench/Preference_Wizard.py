try:
    from scripts import preferences
    from scripts import defaults
except:
    print("Please make sure you are running the proper conda environment")
    exit()
import os
from os import path
from os.path import dirname


def save(save_path):

    # initialize variables
    changed_name = False      # if file has already been used
    files = 0                 # numbers of files that have already been used
    path = os.getcwd()
    file_name = "preferences"
    file_changed = False
    BACKSLASH = "\ ".replace(" ", "")

    # if user included filename
    if ".txt" in save_path:
        temp_path = dirname(save_path)
        file_name = save_path.replace(temp_path + BACKSLASH, "").replace(".txt", "")
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


# Asks the user question to fill out the preferences file
def main():
    # Ask Questions of the user to alter the default preferences
    # Questions should briefly explain what the parameter being changed will effect
    # each question should offer a "more info" command to get an in depth explaination

    # Question 1: Save output path, where any current session will be saved to
    not_answered = True
    while not_answered:
        print("\nWhere would you like the workbench to save the current AI's progress and models?")
        print("\n\tCustom")
        print("\n\tOR")
        print("\n\tDefault")
        print("\n\t\tThe default location is: " + preferences.output)
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower().strip()
        if userInput == "custom":
            pathInput = input("\nPlease provide a system path: ")
            print("\nYou have given the path: ")
            print("\n\t" + pathInput)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                preferences.output = pathInput
                not_answered = False
            else:
                not_answered = True
        elif userInput == "default":
            print("\nThe system will use this default:")
            print("\n\t" + preferences.output)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                not_answered = False
            else:
                not_answered = True
        else:
            print("\nPlease select default or custom")
            not_answered = True

    # Question 2: Saved sessions path, Where a previous session will be stored when the system makes a new session
    not_answered = True
    while not_answered:
        print("\nWhere would you like the workbench to save any previous sessions?")
        print("\n\tCustom")
        print("\n\tOR")
        print("\n\tDefault")
        print("\n\t\tThe default location is: " + preferences.sessions)
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower().strip()
        if userInput == "custom":
            pathInput = input("\nPlease provide a system path: ")
            print("\nYou have given the path: ")
            print("\n\t" + pathInput)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                preferences.sessions = pathInput
                not_answered = False
            else:
                not_answered = True
        elif userInput == "default":
            print("\nThe system will use this default:")
            print("\n\t" + preferences.sessions)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                not_answered = False
            else:
                not_answered = True
        else:
            print("\nPlease select default or custom")
            not_answered = True

    # Question 3: Max Saved sessions, number of sessions that are kept saved oldest one is deleted when max is reached
    not_answered = True
    while not_answered:
        print("\nWhat is the maximum number of saved sessions you want the system to keep before deleting old ones?")
        print("\n\t\tThe default is: " + str(preferences.max_saved_sess))
        userInput = input("\n<WIZARD>: ")
        try:
            userInput = int(userInput)
        except:
            print("\nPlease give an integer value.")
            not_answered = True
            continue
        if userInput <= 0:
            print("Please provide a value greater than 0.")
            not_answered = True
            continue
        print("\nYou have given: ")
        print("\n\t" + str(userInput))
        verify = input("\nIs this correct? (y/n): ")
        if verify == "y":
            preferences.max_saved_sess = userInput
            not_answered = False
        else:
            not_answered = True

    # Question 4: Classifier path, the location of a .names file containing a list of the labels that will be used
    not_answered = True
    while not_answered:
        print("\nWhere would you like the created .name classifier file to be saved?")
        print("\n\tType 'help' for more info")
        print("\n\tType 'default' to use the default location")
        print("\n\t\tThe default location is: " + preferences.classifier_file)
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower().strip()
        if userInput != "help":
            if userInput == 'default':
                pathInput = preferences.classifier_file
            else:
                pathInput = userInput
            print("\nYou have given the path: ")
            print("\n\t" + pathInput)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                preferences.classifier_file = pathInput
                not_answered = False
            else:
                not_answered = True
        else:
            print("\nThe classifier file which has the file type '.names' is a file containing a list of the labels")
            print("\nthat will be used by the AI to identify the data.")
            print("\nThis file is generated automatically from the labels within the data provided to the system.")
            not_answered = True

    # Question 5: Training data location, this should be in the form a .tfrecords
    not_answered = True
    while not_answered:
        print("\nWhere do you want the .tfrecord created from the training data in the images folder to be saved?")
        print("\n\tType 'help' for more info")
        print("\n\tType 'default' to use the default location")
        print("\n\t\tThe default location is: " + preferences.dataset_train)
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower().strip()
        if userInput != "help":
            if userInput == 'default':
                pathInput = preferences.dataset_train
            else:
                pathInput = userInput
            print("\nYou have given the path: ")
            print("\n\t" + pathInput)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                preferences.dataset_train = pathInput
                not_answered = False
            else:
                not_answered = True
        else:
            print("\nThe data in the images folder of the workbench is converted into a .tfrecords file.")
            print("\nThis is a binary based file that greatly speeds up the training when compared to training")
            print("\non a raw image set. This is because Tensorflow must serialize each image for the AI and by doing")
            print("\nit before training starts the system will never need to re-serialize any images it might use again")
            not_answered = True

    # Question 6: Testing data location, this should be in the form a .tfrecords
    not_answered = True
    while not_answered:
        print("\nWhere do you want the .tfrecord created from the testing data in the images folder to be saved?")
        print("\n\tType 'help' for more info")
        print("\n\tType 'default' to use the default location")
        print("\n\t\tThe default location is: " + preferences.dataset_test)
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower().strip()
        if userInput != "help":
            if userInput == 'default':
                pathInput = preferences.dataset_test
            else:
                pathInput = userInput
            print("\nYou have given the path: ")
            print("\n\t" + pathInput)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                preferences.dataset_test = pathInput
                not_answered = False
            else:
                not_answered = True
        else:
            print("\nThe data in the images folder of the workbench is converted into a .tfrecords file.")
            print("\nThis is a binary based file that greatly speeds up the training when compared to training")
            print("\non a raw image set. This is because Tensorflow must serialize each image for the AI and by doing")
            print("\nit before training starts the system will never need to re-serialize any images it might use again")
            not_answered = True

    # Question 7: Validation images location, used to showcase the model at the end of the full process
    not_answered = True
    while not_answered:
        print("\nWhere do you want the images used for validation from the data in the images folder to be saved?")
        print("\n\tType 'help' for more info")
        print("\n\tType 'default' to use the default location")
        print("\n\t\tThe default location is: " + preferences.validate_input)
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower().strip()
        if userInput != "help":
            if userInput == 'default':
                pathInput = preferences.validate_input
            else:
                pathInput = userInput
            print("\nYou have given the path: ")
            print("\n\t" + pathInput)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                preferences.validate_input = pathInput
                not_answered = False
            else:
                not_answered = True
        else:
            print("\nThis folder is used to store the randomly selected files used to validate the models training.")
            print("\nThis validation is used to re-test the model at the end and show you some example results.")
            not_answered = True

    # Question 8: Number of images to validate, the amount of random images pulled and used to showcase
    not_answered = True
    while not_answered:
        print("\nWhat is the number of images you want the system to randomly select for validation?")
        print("\n\t\tThe default is: " + str(preferences.validate_img_num))
        userInput = input("\n<WIZARD>: ")
        try:
            userInput = int(userInput)
        except:
            print("\nPlease give an integer value.")
            not_answered = True
            continue
        if userInput <= 0:
            print("Please provide a value greater than 0.")
            not_answered = True
            continue
        print("\nYou have given: ")
        print("\n\t" + str(userInput))
        verify = input("\nIs this correct? (y/n): ")
        if verify == "y":
            preferences.validate_img_num = userInput
            not_answered = False
        else:
            not_answered = True

    # Question 9: Image Size, the size of the images in the data sets
    not_answered = True
    while not_answered:
        print("\nWhat is the size of the images you are using to create the AI?")
        print("\n\t\tThe default is: " + str(preferences.image_size))
        userInput = input("\n<WIZARD>: ")
        try:
            userInput = int(userInput)
        except:
            print("\nPlease give an integer value.")
            not_answered = True
            continue
        if userInput <= 0:
            print("Please provide a value greater than 0")
            not_answered = True
            continue
        if userInput != 224 and userInput != 256 and userInput != 416:
            print("The workbench only supports 224, 256, and 416 sizes at this time")
            not_answered = True
            continue
        print("\nYou have given: ")
        print("\n\t" + str(userInput))
        verify = input("\nIs this correct? (y/n): ")
        if verify == "y":
            preferences.image_size = userInput
            not_answered = False
        else:
            not_answered = True

    # Question 10: Tiny Yolo or Regular
    not_answered = True
    while not_answered:
        print("\nDo you want to make a Tiny-Yolo AI model or a regular Yolo AI Model?")
        print("\n\tTiny")
        print("\n\tOR")
        print("\n\tRegular")
        print("\n\tOR")
        print("\n\tType 'Help' for more info")
        if preferences.tiny:
            print("\n The default is Tiny-Yolo")
        else:
            print("\n The default is regular Yolo")
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower().strip()
        if userInput == "tiny":
            print("\nYou have set the system for Tiny-Yolo.")
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                preferences.tiny = True
                not_answered = False
            else:
                not_answered = True
        elif userInput == "regular":
            print("\nYou have set the system for regular Yolo.")
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                preferences.tiny = False
                not_answered = False
            else:
                not_answered = True
        elif userInput == "help":
            print("\nYolo is the name of the AI algorithm the workbench is using.")
            print("\nYolo is able to be created in 2 forms: regular and tiny.")
            print("\nTiny Yolo is designed to consume less memory space making it better for direct use in android devices")
            print("\nAt the moment Tiny is experimental within the workbench and is not as well tested as the regular Yolo")
            not_answered = True
        else:
            print("\nPlease select tiny, regular, or help")
            not_answered = True

    # Question 11: Transfer type, chosen from a set of choices
    not_answered = True
    while not_answered:
        print("\nWhat type of transfer do you plan to use to start the models training?")
        print("\n\tNone")
        print("\n\tOR")
        print("\n\tDarknet")
        print("\n\tOR")
        print("\n\tFine_tune")
        print("\n\tOR")
        print("\n\tFrozen")
        print("\n\tOR")
        print("\n\tType 'Help' for more info")
        print("\n")
        print("The default is: " + preferences.transfer)
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower().strip()
        if userInput != "help":
            if userInput != "darknet" \
                    and userInput != "fine_tune" \
                    and userInput != "none" \
                    and userInput != "frozen":
                print("\n Please select one of the given options.")
                not_answered = True
                continue
            print("\nYou have set the system for:")
            print("\n\t" + userInput)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                preferences.transfer = userInput
                not_answered = False
            else:
                not_answered = True
        else:
            print("\nTransfer modes are based on what type of previous AI you are trying to start from.")
            print("\nNone is for when you want to train from scratch and are not using any previous AI.")
            print("\nDarknet is for the use of yolo weights files.")
            print("\nFine_tune is for previous AI checkpoints such as the ones made in the workbench saved as yolov3_train_X.tf")
            print("\nFrozen is for previous Tensorflow 1 models, this mode is very experimental and may not function")
            not_answered = True

    # Question 12: Weights location: Not asked if transfer is set to none, sets Tiny weights instead if Tiny is set
    if preferences.transfer != "none":
        not_answered = True
        while not_answered:
            print("\nWhere is the data you want to transfer for AI use?")
            print("\n\tType 'help' for more info")
            print("\n\tType 'default' to use the default location")
            print("\n\t\tThe default location is: " + preferences.weights)
            print("\n\tNOTICE: This default is intended for 'darknet' transfer")
            userInput = input("\n<WIZARD>: ")
            userInput = str(userInput).lower().strip()
            if userInput != "help":
                if userInput == 'default':
                    pathInput = preferences.weights
                else:
                    pathInput = userInput
                print("\nYou have given the path: ")
                print("\n\t" + pathInput)
                verify = input("\nIs this correct? (y/n): ")
                if verify == "y":
                    preferences.weights = pathInput
                    not_answered = False
                else:
                    not_answered = True
            else:
                print("\nThis should be the exact location of what you wish to transfer.")
                print("\nIf the transfer type is darknet it should be a .weights file.")
                print("\nIf the transfer type is fine_tune it should be a .tf file.")
                print("\nIf the transfer type is frozen it should be a frozen tensorflow model.")
                not_answered = True

    # Question 13: Mode, the mode for training the model, determines early stopping
    not_answered = True
    while not_answered:
        print("\nWhat type of mode of training would you like to use?")
        print("\n\tFit")
        print("\n\tOR")
        print("\n\tEager_fit")
        print("\n\tOR")
        print("\n\tEager_tf")
        print("\n\tOR")
        print("\n\tType 'Help' for more info")
        print("\n")
        print("The default is: " + preferences.mode)
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower().strip()
        if userInput != "help":
            if userInput != "fit" \
                    and userInput != "eager_fit" \
                    and userInput != "eager_tf":
                print("\n Please select one of the given options.")
                not_answered = True
                continue

            print("\nYou have set the system for:")
            print("\n\t" + userInput)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                preferences.mode = userInput
                not_answered = False
            else:
                not_answered = True
        else:
            print("\nThe training mode helps the system to decide how to perfrom early stopping.")
            print("\nEarly stopping is necessary to avoid a common AI issue known as 'overfitting'.")
            print("\nOverfitting is when the AI begins to assume that all aspects of an image are necessary.")
            print("\nThis results in an AI that is not very accurate and that is unable to generalize,")
            print("\nthus preventing it from being able to read unknown data.")
            print("\nFit is the most strict from and will cause the training to stop the moment improvement falters too much.")
            print("\nEager_fit is more lenient and will allow the AI to train a little more, this is useful as ")
            print("\nsometimes AI will have pateaus before improving or may get worse before getting much better ")
            print("\nEager_tf is for system testing purposes only, it is not at all useful for proper AI training")
            not_answered = True
    # Question 14: Epochs, number of times training occurs
    not_answered = True
    while not_answered:
        print("\nHow many iterations of training, known as epochs, do you want?")
        print("\n\tType 'Help' for more info")
        print("\nNOTE: The system may stop training early based on the mode you've previously selected.")
        print("\n\t\tThe default is: " + str(preferences.epochs))
        userInput = input("\n<WIZARD>: ")
        if userInput == 'help':
            print("\nThe number of epochs dictates how many times the model will be trained.")
            print("\nA higher value is a good start unless you are worried that the early stopping will not catch overfitting.")
            continue
        try:
            userInput = int(userInput)
        except:
            print("\nPlease give an integer value.")
            not_answered = True
            continue
        if userInput <= 0:
            print("Please provide a value greater than 0")
            not_answered = True
            continue
        print("\nYou have given: ")
        print("\n\t" + str(userInput))
        verify = input("\nIs this correct? (y/n): ")
        if verify == "y":
            preferences.epochs = userInput
            not_answered = False
        else:
            not_answered = True
    # Question 14: Max Checkpoints, number of checkpoints saved,
    #   each epoch tries to save a checkpoint which take up about 1gig
    not_answered = True
    while not_answered:
        print("\nEach epoch saves a checkpoint to shows its training, what is the max amount you want saved?")
        print("\nNOTE: Each checkpoint is about 1 gigabyte of memory.")
        print("\n\t\tThe default is: " + str(preferences.max_checkpoints))
        userInput = input("\n<WIZARD>: ")
        try:
            userInput = int(userInput)
        except:
            print("\nPlease give an integer value.")
            not_answered = True
            continue
        if userInput <= 0:
            print("Please provide a value greater than 0")
            not_answered = True
            continue
        print("\nYou have given: ")
        print("\n\t" + str(userInput))
        verify = input("\nIs this correct? (y/n): ")
        if verify == "y":
            preferences.max_checkpoints = userInput
            not_answered = False
        else:
            not_answered = True
    # Question 15: Batch Size, Amount of images randomly selected to train and test the AI
    #   The higher this is and the larger the images the more RAM consumed, if the consumption exceeds available RAM
    #   then the program will crash before training occurs
    not_answered = True
    while not_answered:
        print("\nWhat is the batch size of images you would lke to load for each training?")
        print("\nNOTE: Batch size consumes a lot of RAM, but the higher the better.")
        print("\n\t\tThe default is: " + str(preferences.batch_size))
        userInput = input("\n<WIZARD>: ")
        if userInput == 'help':
            print("\nBatch size dictates the number of images loaded into your system RAM for an epoch of training.")
            print("\nSince the system will go through all images provided an increase in batch size will lower the")
            print("\namount of times the system will ned to load new images in and thus speed up the training")
            continue
        try:
            userInput = int(userInput)
        except:
            print("\nPlease give an integer value.")
            not_answered = True
            continue

        if userInput <= 0:
            print("Please provide a value greater than 0")
            not_answered = True
            continue

        print("\nYou have given: ")
        print("\n\t" + str(userInput))
        if userInput > 16 :
            print("\nWARNING: This is a fairly high batch_size for an average computer.")
            print("\n If you do not have enough RAM the system may crash.")

        verify = input("\nIs this batch size correct? (y/n): ")
        if verify == "y":
            preferences.batch_size = userInput
            not_answered = False
        else:
            not_answered = True

    # Question 16: Name of this .txt preference file
    not_answered = True
    name = ""
    while not_answered:
        print("\nWhat would you like to name this file?")
        userInput = input("\n<WIZARD>: ")
        name = userInput + ".txt"
        print("\nThis preference file will save as: ")
        print("\n\t" + name)
        verify = input("\nIs this correct? (y/n): ")
        if verify == "y":
            not_answered = False
        else:
            not_answered = True
    # Question 17: Location to save this file
    not_answered = True
    save_path = name
    while not_answered:
        print("\nWhere would you like to save this file?")
        print("\n\tCustom")
        print("\n\tOR")
        print("\n\tDefault")
        print("\n\t\tThe default location is: " + os.getcwd())
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower().strip()
        if userInput == "custom":
            pathInput = input("\nPlease provide a system path: ")
            print("\nThe file and save location is: ")
            print("\n\t" + pathInput + save_path)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                save_path = pathInput + save_path
                not_answered = False
            else:
                not_answered = True
        elif userInput == "default":
            print("\nThe system will use this default:")
            print("\n\t" + os.getcwd() + "\\" + save_path)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                pathInput = os.getcwd()
                save_path = pathInput + "\\" + save_path
                not_answered = False
            else:
                not_answered = True
        else:
            print("\nPlease select default or custom")
            not_answered = True
    print("Saving...")
    save(save_path)

main()
