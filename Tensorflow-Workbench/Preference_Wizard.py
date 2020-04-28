from scripts import preferences
from scripts import defaults
import os


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
        userInput = str(userInput).lower()
        if userInput == "custom":
            pathInput = input("\nPlease provide a system path: ")
            preferences.output = pathInput
            print("\nYou have given the path: ")
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
        userInput = str(userInput).lower()
        if userInput == "custom":
            pathInput = input("\nPlease provide a system path: ")
            preferences.sessions = pathInput
            print("\nYou have given the path: ")
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
        print("\n\t\tThe default is: " + preferences.max_saved_sess)
        userInput = input("\n<WIZARD>: ")
        try:
            userInput = int(userInput)
        except:
            print("\nPlease give an integer value.")
            pass
        preferences.max_saved_sess = userInput
        print("\nYou have given: ")
        print("\n\t" + preferences.max_saved_sess)
        verify = input("\nIs this correct? (y/n): ")
        if verify == "y":
            not_answered = False
        else:
            not_answered = True

    # Question 4: Classifier path, the location of a .names file containing a list of the labels that will be used
    not_answered = True
    while not_answered:
        print("\nWhere is the classifier .names file you plan to use located?")
        print("\n\tType 'help' for more info")
        print("\n\t\tThe default location is: " + preferences.classifier_file)
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower()
        if userInput != "help":
            pathInput = userInput
            preferences.classifier_file = pathInput
            print("\nYou have given the path: ")
            print("\n\t" + preferences.classifier_file)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                not_answered = False
            else:
                not_answered = True
        else:
            print("\nThe classifier file which has the file type '.names' is a file containing a list of the labels")
            print("\nthat will be used by the AI to identify the data.")
            print("\n\n\tThis label list should include every label that will be present in the datasets you use")
            not_answered = True

    # Question 5: Training data location, this should be in the form a .tfrecords
    not_answered = True
    while not_answered:
        print("\nWhere do you want the .tfrecord created from the data in the images folder to be saved?")
        print("\n\tType 'help' for more info")
        print("\n\t\tThe default location is: " + preferences.dataset_train)
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower()
        if userInput != "help":
            pathInput = userInput
            preferences.dataset_train = pathInput
            print("\nYou have given the path: ")
            print("\n\t" + preferences.dataset_train)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
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
        print("\nWhere do you want the .tfrecord created from the data in the images folder to be saved?")
        print("\n\tType 'help' for more info")
        print("\n\t\tThe default location is: " + preferences.dataset_test)
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower()
        if userInput != "help":
            pathInput = userInput
            preferences.dataset_test = pathInput
            print("\nYou have given the path: ")
            print("\n\t" + preferences.dataset_test)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
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
        print("\nWhere do you want the .tfrecord created from the data in the images folder to be saved?")
        print("\n\tType 'help' for more info")
        print("\n\t\tThe default location is: " + preferences.validate_input)
        userInput = input("\n<WIZARD>: ")
        userInput = str(userInput).lower()
        if userInput != "help":
            pathInput = userInput
            preferences.validate_input = pathInput
            print("\nYou have given the path: ")
            print("\n\t" + preferences.validate_input)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
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
        print("\n\t\tThe default is: " + preferences.validate_img_num)
        userInput = input("\n<WIZARD>: ")
        try:
            userInput = int(userInput)
        except:
            print("\nPlease give an integer value.")
            pass
        preferences.validate_img_num = userInput
        print("\nYou have given: ")
        print("\n\t" + preferences.validate_img_num)
        verify = input("\nIs this correct? (y/n): ")
        if verify == "y":
            not_answered = False
        else:
            not_answered = True

    # Question 9: Image Size, the size of the images in the data sets
    not_answered = True
    while not_answered:
        print("\nWhat is the size of the images you are using to create the AI?")
        print("\n\t\tThe default is: " + preferences.image_size)
        userInput = input("\n<WIZARD>: ")
        try:
            userInput = int(userInput)
        except:
            print("\nPlease give an integer value.")
            pass
        preferences.image_size = userInput
        print("\nYou have given: ")
        print("\n\t" + preferences.image_size)
        verify = input("\nIs this correct? (y/n): ")
        if verify == "y":
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
        userInput = str(userInput).lower()
        if userInput == "tiny":
            preferences.tiny = True
            print("\nYou have set the system for Tiny-Yolo.")
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                not_answered = False
            else:
                not_answered = True
        elif userInput == "regular":
            preferences.tiny = False
            print("\nYou have set the system for regular Yolo.")
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
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
        userInput = str(userInput).lower()
        if userInput != "help":
            preferences.transfer = userInput
            if userInput is not "darknet" or "fine_tune" or "none" or "frozen":
                print("\n Please select one of the given options.")
                not_answered = True
                pass

            print("\nYou have set the system for:")
            print("\n\t" + preferences.transfer)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
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
    if preferences.transfer is not "none":
        not_answered = True
        while not_answered:
            print("\nWhere is the data you want to transfer for AI use?")
            print("\n\tType 'help' for more info")
            print("\n\t\tThe default location is: " + preferences.weights)
            userInput = input("\n<WIZARD>: ")
            userInput = str(userInput).lower()
            if userInput != "help":
                pathInput = userInput
                preferences.validate_input = pathInput
                print("\nYou have given the path: ")
                print("\n\t" + preferences.validate_input)
                verify = input("\nIs this correct? (y/n): ")
                if verify == "y":
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
        userInput = str(userInput).lower()
        if userInput != "help":
            preferences.transfer = userInput
            if userInput is not "fit" or "eager_fit" or "eager_tf":
                print("\n Please select one of the given options.")
                not_answered = True
                pass

            print("\nYou have set the system for:")
            print("\n\t" + preferences.mode)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
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
        print("\nNOTE: The system may stop training early based on the mode you've previously selected.")
        print("\n\t\tThe default is: " + preferences.epochs)
        userInput = input("\n<WIZARD>: ")
        try:
            userInput = int(userInput)
        except:
            print("\nPlease give an integer value.")
            pass
        preferences.epochs = userInput
        print("\nYou have given: ")
        print("\n\t" + preferences.epochs)
        verify = input("\nIs this correct? (y/n): ")
        if verify == "y":
            not_answered = False
        else:
            not_answered = True
    # Question 14: Max Checkpoints, number of checkpoints saved,
    #   each epoch tries to save a checkpoint which take up about 1gig
    not_answered = True
    while not_answered:
        print("\nEach epoch saves a checkpoint to shows its training, what is the max amount you want saved?")
        print("\nNOTE: Each checkpoint is about 1 gigabyte of memory.")
        print("\n\t\tThe default is: " + preferences.max_checkpoints)
        userInput = input("\n<WIZARD>: ")
        try:
            userInput = int(userInput)
        except:
            print("\nPlease give an integer value.")
            pass
        preferences.max_checkpoints = userInput
        print("\nYou have given: ")
        print("\n\t" + preferences.max_checkpoints)
        verify = input("\nIs this correct? (y/n): ")
        if verify == "y":
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
        print("\n\t\tThe default is: " + preferences.batch_size)
        userInput = input("\n<WIZARD>: ")
        try:
            userInput = int(userInput)
        except:
            print("\nPlease give an integer value.")
            pass
        preferences.batch_size = userInput
        print("\nYou have given: ")
        print("\n\t" + preferences.batch_size)
        if(preferences.batch_size > 16):
            print("\nWARNING: This is a fairly high batch_size for an average computer.")
            print("\n If you do not have enough RAM the system may crash.")

        verify = input("\nIs this batch size correct? (y/n): ")
        if verify == "y":
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
        userInput = str(userInput).lower()
        if userInput == "custom":
            pathInput = input("\nPlease provide a system path: ")
            save_path = pathInput + save_path
            print("\nThe file and save location is: ")
            print("\n\t" + save_path)
            verify = input("\nIs this correct? (y/n): ")
            if verify == "y":
                not_answered = False
            else:
                not_answered = True
        else:
            print("\nPlease select default or custom")
            not_answered = True
    print("Saving...")
    save(save_path)

main()
