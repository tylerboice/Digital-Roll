import os
import random
import shutil
import time
import math
import xml.etree.ElementTree as ET
import base64

from os import path

# Global variables
unlabelled_files = []                # all imgs found that don't have correspoding xml
temp_classifiers = []                # temporary classifiers list
CHECKPOINT_KEYWORD = "yolov3_train_" # prefix to all checkpoints
ERROR = "ERROR_MESSAGE"              # Error message value
IMAGE_TYPES = ['.png', '.jpeg', '.jpg']
XML_TYPE = '.xml'


########################## CHECK FILES EXIST #############################
# Description: ensures there are images in image dir and checks if all directories exists
#              if not it creates the directory, unless weights file does not exist, then fail
# Parameters: image_path - String - path to the folder with all the  images
#             data_path -  String - path to the data directory
#             min_images - Int - minimum amonut of images to run the workbench
#             output_path - String - path to ouput files from the workbench
#             save_sess - String - path to save previous sessions
#             test_image_path - String - sub folder of image path that will contain testing images
#             train_image_path - String - sub folder of image path that will contain training images
#             val_image_path - String - sub folder of image path that will contain to validatae after model created
#             weights_path = String - patht o weights file to train model from
# Return: Number of images, unless weights path does not exist, then return false
def check_files_exist(image_path, data_path, min_images, output_path, save_sess,
                      test_image_path, train_image_path, val_image_path, weights_path, trans):

    image_path = (image_path + "/").replace("//", "/")

    ####### IMAGE PATH #######
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    # Check if images in image_path
    extract_img_from_xml(image_path)
    extract_sub_dirs(image_path, image_path)
    remove_empty_folders(image_path)

    images_found = get_img_count(image_path)

    # print unlabelled files
    if len(unlabelled_files) != 0:
        print("Images not labelled = " + str(len(unlabelled_files)))
        print("\nThe following images do not have an xml file:")
        for item in unlabelled_files:
            print("\t" + item)

    # if images found
    if images_found != 0:
        ####### DATA PATH #######
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        ####### OUTPUT MODEL PATH #######
        if not os.path.exists(save_sess):
            os.mkdir(save_sess)

        ####### SAVED SESS PATH #######
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        ####### TEST IMAGE PATH #######
        if not os.path.exists(test_image_path):
            os.mkdir(test_image_path)

        ####### TRAIN IMAGE PATH #######
        if not os.path.exists(train_image_path):
            os.mkdir(train_image_path)

        ####### VALIDATE IMAGE PATH #######
        if not os.path.exists(val_image_path):
            os.mkdir(val_image_path)

        ####### WEIGHTS PATH #######
        if weights_path != None:
            if not os.path.exists(weights_path) and trans == "darknet":
                print("\nERROR: The weights path does not exist")
                print("\n\tWeights Path Location: " + weights_path)
                print("\n\tDownload the yolo3_weight file at https://pjreddie.com/media/files/yolov3.weights")
                return False

    return images_found


######################### CLASSIFER REMOVE DUP ####################################
# Description: removes all duplicates from classifier list and prints the count of each one
# Parameters: classifiers - List - a list with all classifiers labelled in image folder (including duplicates)
# Return: new_list - List - classifiers list with duplicates removed
def classifier_remove_dup(classifiers):

    print("\n\tClassifiers:")
    total_classifiers = 0
    new_list = []

    # iterate though classifiers and add only unique claddifiers to new_list, then sort
    classes = 0
    while classes < len(classifiers):
        if classifiers[classes] not in new_list:
            new_list.append(classifiers[classes])
        classes += 1
    new_list.sort()

    # for every unqiue classifier in new_list, print the amount of that classifer found in classifiers(input)
    classes = 0
    while classes < len(new_list):
        print("\t\t" + str(new_list[classes]) + ": " + str(classifiers.count(new_list[classes])))
        total_classifiers += classifiers.count(new_list[classes])
        classes += 1
    print("\n\tTotal Objects: " + str(total_classifiers))

    return new_list

########################## CONVERT TO TIME #############################
# Description: takes in float and converts it to time
# Parameters: time - FLOAT - time
# Return: output - String - string of time
def convert_to_time(time):
    seconds = round(time)
    minutes = math.floor(time/60)
    hours = math.floor(minutes/60)
    output = ""
    if hours > 1:
        output += str(hours) + " Hours, "
        minutes %= 60
    elif hours == 1:
        output += str(hours) + " Hour, "
        minutes %= 60

    if minutes > 1:
        output += str(minutes) + " Minutes, "
        seconds %= 60
    elif minutes == 1:
        output += str(minutes) + " Minute, "
        seconds %= 60

    if seconds == 1:
        output += str(minutes) + " Second"
    else:
        output += str(seconds) + " Seconds"
    return output


########################## CREATE CLASSIFIER FILE #############################
# Description: creates classifier file, and writes each classifier on a new line in the file
# Parameters: file - String - name of classifier file that will be created
#            classifiers - List - list of unique classifiers
# Return: None
def create_classifier_file(file, classifiers):

    class_counter = 0
    with open(file, "w") as f:
        for classification in classifiers:
            class_counter += 1
            f.write(classification + "\n")


############################ DUPLICATE PB ###########################
# Does not currently solve issue found when creating models for mobile applications
'''
def duplicate_pb(path):
    tf_model = "NOT_FOUND"
    path = (path + "/").replace("//", "/")
    temp_folder = path + "temp_folder/"
    for file in os.listdir(path):
        if ".pb" in file:
            tf_model = path + file
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.mkdir(temp_folder)
    shutil.copy(tf_model, temp_folder + "saved_model.pb")
    return temp_folder
'''

###################### EXTRACT IMG FROM XML #############################
# Description: extractes and image that is embedded using base64 within the xml file
# Parameters: path - String - xml file/path
# Return: None
def extract_img_from_xml(path):

    # if path given is directory, recurse inside directory
    if os.path.isdir(path):
        path = (path + "/").replace("//", "/")
        for file in os.listdir(path):
            extract_img_from_xml(path + file)

    # if xml file found
    elif XML_TYPE in path:
        tree = ET.parse(path)
        root = tree.getroot()

        # find the img tag and decode from base 64 to .jpg
        for img in root.findall('img'):
            base64_str = img.text.replace('Optional("', "").replace('=")', "")
            base64_remainder = len(base64_str) % 4
            while base64_remainder > 0:
                base64_str += "="
                base64_remainder -= 1
            filename = path.replace(XML_TYPE, ".jpg")
            decoded_str = base64.b64decode(base64_str)

            # create .jpg
            with open(filename, 'wb') as f:
                f.write(decoded_str)


########################## EXTRACT SUB DIRS #############################
# Description: extractes all xml and images within the subfolders, this is only used when images
#              are found in the imedate image folder. This is because it means images were added by user so all images
#              and xml will be extracted from subfolder and placed in the image folder
# Parameters: image_path - String - path where the images and xmls are
#             current_path - String - used to determine current path during recurssion
# Return: None
def extract_sub_dirs(image_path, current_path):

    current_path = (current_path + "/").replace("//", "/")
    new_images = False

    # search for images in the directory
    for filename in os.listdir(current_path):
        filename = filename.lower()
        if filename.endswith(tuple(IMAGE_TYPES)):
            new_images = True

    # if new images found, then extract all images and xml from sub-dir and place in image_path
    if new_images:
        for filename in os.listdir(current_path):
            filename = filename.lower()
            if os.path.isdir(current_path + filename):
                extract_sub_dirs(image_path, current_path + filename)
            elif filename.endswith(tuple(IMAGE_TYPES)) or filename.endswith(XML_TYPE):
                shutil.move(current_path + filename, image_path + filename)


########################## FROM WORKBENCH #############################
# Description: removes all directories from current working directory so it starts at the Workbench
#              example: input: C:Users/John/Desktop/Digital_roll/Tensorflow-Workbench/scripts
#                       output: ./scripts
# Parameters: path - String - current working directory
# Return: the path with directoryies removed
def from_workbench(path):
    keyword = "Tensorflow-Workbench"
    try:
        if keyword in path:
            return "." + path.split(keyword)[1]
        else:
            return path
    except:
        return str(path)


############################ GET NEWEST CHECKPOINT ###############################
# Description: find the checkpoint that has the newest "last_modified" time
#              This is used because epochs loop though and overwrite, so if epochs were
#              canceled by user or stopped, then a lower number epoch may be newer then a
#              epoch with a higher value
# Parameters: checkpoint_path - String - path to checkpoints
# Return: the newest cehckpoint file
def get_newest_checkpoint(checkpoint_path):
    newest = -1
    checkpoint = checkpoint_path
    newest_file = ""

    # loop though checkpoints
    for file in os.listdir(checkpoint_path):
        if CHECKPOINT_KEYWORD in file:

            # get the time stamp and place in format: int(year+month+day+time)
            last_modified = time.ctime(os.path.getmtime(checkpoint + file)).split(' ')
            str_last_mod = str(last_modified[4]) + str(get_month(last_modified[1])) + str(last_modified[3])
            last_modified = int(str_last_mod.replace(":", ""))

            # if current checkpoint is older, set to newest file
            if last_modified > newest:
                newest_file = file
                newest = last_modified
    return newest_file


############################ GET CHECKPOINT INT ###############################
# Description: helper method that takes a checkpoint file and returns the integer portion of the name
#              example: input = yolov3_train_23.tf.index
#                       output = 23
# Parameters: file - String - checkpoint file
# Return: integer value within the checkpoint name
def get_checkpoint_int(file, checkpoint_path):
    try:
        file = file.replace(checkpoint_path, "")
        file = file.replace(CHECKPOINT_KEYWORD, "").split(".")[0]
        if not file.isnumeric() and "/" in file:
            file = file.split("/")[:-1]
        return int(file)
    except:
        return 0


########################## GET CLASSIFIERS #############################
# Description: searchs though all xml files and append the classifiers within each file to a list
# Parameters: path - String - path of xml files
# Return: list with all classifiers found, duplicates included
def get_classifiers(path):
    class_counter = 0
    classifiers = []

    # if path exists
    if os.path.isdir(path):
        data_dir = (path + "/").replace("//", "/")

        # recurse into subfolders, append to classifiers list
        for file in os.listdir(path):
            nested_folder = get_classifiers(path + "/" + file)
            if classifiers == [] or classifiers == None:
                classifiers = nested_folder
            else:
                classifiers = classifiers + nested_folder

    # if path is .xml extract classifiers and append to classifiers list
    if XML_TYPE in path:
        tree = ET.parse(path)
        root = tree.getroot()
        for object in root.findall('object'):
            for name in object.findall('name'):
                classifiers.append(name.text)
    return classifiers


########################## GET IMG COUNT #############################
# Description: searchs though path and counts all the jpg and xml files
# Parameters: path - String - path of image and xml files
# Return: total_images - Int - count of total images found
def get_img_count(path):
    path = path.lower()
    total_images = 0

    # path is directory - For every file image in the path, check if it has an xml file and move it
    if os.path.isdir(path):
        path = (path + "/").replace("//", "/")

        # recurse intoo sub directory
        for filename in os.listdir(path):
            total_images += get_img_count(path + filename)

    # file is image, add it to count and check if it has a xml file of the same name
    if path.endswith(tuple(IMAGE_TYPES)):
        path_type = get_type(path)
        found_label = False
        xml_version = path.split(path_type)[0] + XML_TYPE
        total_images += 1

        # xml found
        if os.path.exists(xml_version):
            found_label = True

        # xml not found
        if found_label == False:
            unlabelled_files.append(path)
    return total_images


########################## GET_INPUT_VALUE ###########################
# Description: extracts the value from a string. Used for getting lines for preference file
#              example: input  = batch_size: 32
#                       output = 32
# Parameters: input - String - line of input from preference file
#             split_char - Char - charter that seperates variable name and value
# Return: the value given for the input
def get_input_value(input, split_char):
    if split_char in input:
        input = input.split(split_char)
        input = ''.join(input[1:])
    return input.strip()



########################## GET_INPUT_VAR ###########################
# Description: extracts the variable from a string. Used for getting lines for preference file
#              example: input = batch_size: 32
#                       output = batch_size
# Parameters: input - String - line of input from preference file
#             split_char - Char - charter that seperates variable name and value
# Return: the varibale given for the input
def get_input_var(input, split_char):
    if split_char in input:
        input = input.split(split_char)[0]
        input = input.strip()
    return input


########################## GET LAST CHECKPOINT #############################
# Description: gets checkpoint with the highest number
# Parameters: checkpoint_path - String - path of the checkpoints
# Return: last_checkpoint - String - checkpoint with the highest number
#         returns ERROR message if bad input was given
def get_last_checkpoint(checkpoint_path):
    last_checkpoint_num = -1
    last_checkpoint = ""
    if os.path.exists(checkpoint_path):
        for filename in os.listdir(checkpoint_path):
            if os.path.isdir(filename):
                temp_check = get_last_checkpoint(filename)
                if  last_checkpoint_num < get_checkpoint_int(temp_check, checkpoint_path):
                    last_checkpoint = checkpoint_path + temp_check

            # if filename is a checkpoint, check to see it is the highest value
            if CHECKPOINT_KEYWORD and '.tf.index' in filename:
               if last_checkpoint_num < get_checkpoint_int(filename, checkpoint_path):
                   last_checkpoint_num = get_checkpoint_int(filename, checkpoint_path)
                   last_checkpoint = checkpoint_path + filename

    # return checkpoint if valid
    if last_checkpoint_num == -1:
        last_checkpoint = ERROR
    return last_checkpoint

########################## GET LOWEST CHECKPOINT #############################
# Description: gets checkpoint with the highest number
# Parameters: checkpoint_path - String - path of the checkpoints
# Return: loest_checkpoint - String - checkpoint with the lowestest number
#         returns ERROR message if bad input was given
def get_lowest_checkpoint(checkpoint_path, counter):
    lowest_checkpoint_num = 999999
    lowest_checkpoint = ""
    if os.path.exists(checkpoint_path):
        for filename in os.listdir(checkpoint_path):
            if os.path.isdir(filename):
                temp_check = get_lowest_checkpoint(filename, counter)
                temp_check_int = get_checkpoint_int(temp_check, checkpoint_path)
                if  lowest_checkpoint_num > temp_check_int and temp_check_int >= counter:
                    lowest_checkpoint = checkpoint_path + temp_check

            # if filename is a checkpoint, check to see it is the highest value
            if CHECKPOINT_KEYWORD and '.tf.index' in filename:
               cur_check_int = get_checkpoint_int(filename, checkpoint_path)
               if lowest_checkpoint_num > cur_check_int and cur_check_int >= counter:
                   lowest_checkpoint_num = get_checkpoint_int(filename, checkpoint_path)
                   lowest_checkpoint = checkpoint_path + filename

    # return checkpoint if valid
    if lowest_checkpoint_num == 999999:
        lowest_checkpoint = ERROR
    return lowest_checkpoint_num

############################ GET MONTH ###############################
# Description: converts month abreviation into int
# Parameters: month - String - three letter month abreviation
# Return:  Int - integer representing month
def get_month(month):
    month = str(month).lower()
    if month == 'jan':
        return 1
    elif month == 'feb':
        return 2
    elif month == 'mar':
        return 3
    elif month == 'apr':
        return 4
    elif month == 'may':
        return 5
    elif month == 'jun':
        return 6
    elif month == 'jul':
        return 7
    elif month == 'aug':
        return 8
    elif month == 'sep':
        return 9
    elif month == 'oct':
        return 10
    elif month == 'nov':
        return 11
    elif month == 'dec':
        return 12
    return 0


########################## GET NUM CLASSES ##########################
# Description: gets the number of classes given the .names file
# Parameters: file - String - .names file
# Return:  Int - number of classifers
def get_num_classes(file):
    num_classes = 0

    # create file if does not exist
    if not os.path.exists(file):
        open(file, "w")

    # count lines in file
    with open(file, "r") as f:
        for line in f.readlines():
            if len(line.strip()) != 0 :
                num_classes = num_classes + 1
    return num_classes


########################## GET OUTPUT FILE #############################
# Description: gets the number of classes given the .names file
# Parameters: file - String - .names file
# Return:  Int - number of classifers
def get_output_file(output):
    file = "workbench_log.txt"
    file_count = 1
    file_path = output + file
    if not os.path.exists(output):
        os.mkdir(output)
    while os.path.exists(file_path):
        file_path = output + "workbench_log-" + str(file_count) + ".txt"
        file_count += 1
    return file_path


def get_type(file):
    if file.endswith(tuple(IMAGE_TYPES)):
        file = "." + file.split('.')[-1]
    return file

############################ FILE IS VALID ###########################
# Returns true if the file exists and is not empty, else false
def is_valid(file):
    return os.path.exists(file) and os.stat(file).st_size != 0


########################## REMOVE EMPTY FOLDERS #############################
# recursivly checks driectory for images
def remove_empty_folders(path):
    for file in os.listdir(path):
        if os.path.isdir(path + file):
            file_path = path + file
            if len(os.listdir(file_path)) == 0:
                os.rmdir(file_path)


########################## REMOVE SMALLEST SESS ##########################
def remove_smallest_sess(save_sess_path, max_saved_sess, keyword):
    session_values = []
    for file in os.listdir(save_sess_path):
        if keyword in file:
            if file.split('-')[1].isnumeric():
                session_values.append(int(file.split('-')[1]))

    # get oldest value
    while len(session_values) >= max_saved_sess:
        oldest_file = session_values[0]
        for sess in session_values:
            if sess < oldest_file:
                oldest_file = sess
        oldest_sess = save_sess_path + keyword + str(oldest_file)
        shutil. rmtree(oldest_sess)
        session_values.remove(oldest_file)
        print("\tOldest session: " + from_workbench(oldest_sess) + " was removed\n")

    # get newest value
    if len(session_values) > 0:
        newest = session_values[0]
        for sess in session_values:
            if sess > newest:
                newest = sess
        return newest
    else:
        return 0

def find_max_checkpoint(path):
    max_checkpoint = 0
    for file in os.listdir(path):
        if CHECKPOINT_KEYWORD in file and ".tf.index" in file:
            max_checkpoint += 1
    return max_checkpoint


def get_checkpoint_suffix(file):
    if ".tf" in file:
        return ".tf" + file.split(".tf")[1]
    return file


############################ REANAME CHECKPOINTS ###############################
# Example: yolov3_train_1.tf.index
# returns: .tf.index
def rename_checkpoints(checkpoint_path):
    max_checkpoints = find_max_checkpoint(checkpoint_path) # max checkpoints
    newest_file = get_newest_checkpoint(checkpoint_path)   # path of the checkpoint
    newest_file_int = get_checkpoint_int(newest_file, checkpoint_path)      # value of the checkpoint
    highest_check_int = get_checkpoint_int(get_last_checkpoint(checkpoint_path), checkpoint_path)

    counter = 1
    if highest_check_int != newest_file_int:

        # rename all the older files
        while counter <= newest_file_int:
            for file in os.listdir(checkpoint_path):
                if CHECKPOINT_KEYWORD + str(counter) + ".tf" in file and not os.path.isdir(file):
                    old_file = checkpoint_path + file
                    new_file = checkpoint_path + CHECKPOINT_KEYWORD + str(counter + highest_check_int) + get_checkpoint_suffix(file)

                    os.rename(old_file, new_file)
            counter += 1

    # while the highest checkpoint doesnt equal max_checkpoints
    counter = 1
    if max_checkpoints !=  highest_check_int:
        while counter <= max_checkpoints:
            lowest_check = get_lowest_checkpoint(checkpoint_path, counter)
            for file in os.listdir(checkpoint_path):
                if CHECKPOINT_KEYWORD + str(lowest_check) in file:
                    old_file = checkpoint_path + file
                    new_file = checkpoint_path + CHECKPOINT_KEYWORD + str(counter) + get_checkpoint_suffix(file)
                    os.rename(old_file, new_file)
            counter += 1


########################## SAVE CHECKPOINTS ##########################
def save_session(default_output, checkpoint_output, save_sess_path, max_saved_sess):
    keyword = "saved_session-"
    contents = False
    if not os.path.exists(save_sess_path):
        os.mkdir(save_sess_path)
    current_sess = remove_smallest_sess(save_sess_path, max_saved_sess, keyword) + 1
    current_sess = (save_sess_path + "/" + keyword + str(current_sess)).replace("\\", "/")
    current_sess = current_sess.replace("//", "/")
    for file in os.listdir(checkpoint_output):
        if CHECKPOINT_KEYWORD in file or ".pb" in file or ".mlmodel" in file:
            contents = True
    if contents:
        os.mkdir(current_sess)
        print("\tPrevious session stored in: " + from_workbench(current_sess) + "\n")
        if os.path.exists(checkpoint_output):
            for file in os.listdir(checkpoint_output):
                shutil.move(checkpoint_output + file, current_sess)


########################## SORT IMAGES #############################
# Takes all the images in the image folder and places them in test, train and validate
# Train = 90% of the images
# Test = 10% of the images
# Validate = takes user spicifed amount of files out of train (num_validate)
def sort_images(num_validate, image_path, test_image_path, train_image_path, val_image_path, total_images):

    # Method Variables
    current_image = 0
    train_images = 0
    valid_images = []
    get_valid = False

    # For every file image in the image dir, check if it has an xml file and move it
    for filename in os.listdir(image_path):
        filename = filename.lower()
        if filename.endswith(tuple(IMAGE_TYPES)):
            found_label = False
            xml_version = filename.split(".")[0] + XML_TYPE
            current_image += 1

            # move to test
            if (current_image % 5 == 0) or (total_images < 10 and current_image == total_images):
                if path.exists(image_path + xml_version):
                    if not path.exists(test_image_path + filename) and not path.exists(test_image_path + xml_version):
                        shutil.move(image_path + filename, test_image_path)
                        shutil.move(image_path + xml_version, test_image_path)
                        found_label = True

            # move to train
            else:
                if path.exists(image_path + xml_version):
                    if not path.exists(train_image_path + filename) and not path.exists(train_image_path + xml_version):
                        shutil.move(image_path + filename, train_image_path)
                        shutil.move(image_path + xml_version, train_image_path)
                        found_label = True


    # count all image and .xml files in train
    for filename in os.listdir(train_image_path):
        filename = filename.lower()
        if filename.endswith(tuple(IMAGE_TYPES)):
            train_images += 1

    # move all files in validate to train
    for file in os.listdir(val_image_path):
        if not path.exists(train_image_path + file):
            shutil.move(val_image_path + file, train_image_path)

    # gather all valid images from train
    if train_images - 1 > num_validate:
        while len(valid_images) < num_validate:
            next_valid = random.randint(1, train_images)
            if next_valid not in valid_images:
                valid_images.append(next_valid)

    # move random valid images from train to validate
    file_count = 0
    for file in os.listdir(train_image_path):
        file = file.lower()
        if file.endswith(tuple(IMAGE_TYPES)):
            file_count += 1
            xml_version = file.split(".")[0] + XML_TYPE
            if file_count in valid_images:
                shutil.move(train_image_path + file, val_image_path)
                if path.exists(train_image_path + xml_version):
                    shutil.move(train_image_path + xml_version, val_image_path)


##################### WRITE TO CHECKPOINT FILE ########################
def write_to_checkpoint(checkpoint_name, filename):
    quote = '"'
    if CHECKPOINT_KEYWORD in checkpoint_name:
        checkpoint_name = CHECKPOINT_KEYWORD + str(checkpoint_name.split(CHECKPOINT_KEYWORD)[1])
        models = "model_checkpoint_path: "
        all_models = "all_model_checkpoint_paths: "
        with open(filename, "w") as f:
            f.write(models + quote + checkpoint_name + quote)
            f.write("\n")
            f.write(all_models + quote + checkpoint_name + quote)
