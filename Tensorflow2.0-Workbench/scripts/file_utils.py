import os
import random
import shutil
import time
import xml.etree.ElementTree as ET
import base64

from os import path

# Global variables
unlabelled_files = []                # all imgs found that don't have correspoding xml
temp_classifiers = []
CHECKPOINT_KEYWORD = "yolov3_train_" # prefix to all checkpoints
ERROR = "ERROR_MESSAGE"              # Error message value



########################## Checking FOR FILES #############################
# checks if all necessary files exist
def checkIfNecessaryPathsAndFilesExist(image_path, data_path, min_images, output_path, save_sess,
                                     test_image_path, train_image_path, val_image_path, weights_path):

    image_path = (image_path + "/").replace("//", "/")

    ####### IMAGE PATH #######
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    extract_img_from_xml(image_path)
    extract_sub_dirs(image_path, image_path)
    remove_empty_folders(image_path)
    images_found = check_for_images(image_path, min_images)
    if images_found != 0:
        ####### DATA PATH #######
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        ####### OUTPUT MODEL PATH #######
        if not os.path.exists(save_sess):
            os.mkdir(save_sess)

        ####### SAVED SESS  PATH #######
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

        if not os.path.exists(weights_path):
            print("\nERROR: The weights path does not exist")
            print("\n\tWeights Path Location: " + weights_path)
            print("\n\tDownload the yolo3_weight file at https://pjreddie.com/media/files/yolov3.weights")
            return False

    return images_found

########################## CHECK FOR IMAGES #############################
# checks a directory for images
def check_for_images(path, min_images):

    total_images = get_img_count(path)
    # print total image count
    if len(unlabelled_files) != 0:
        print("Images not labelled = " + str(len(unlabelled_files)))
    if len(unlabelled_files) != 0:
        print("\nThe following images do not have an xml file:")
        for item in unlabelled_files:
            print("\t" + item)

    return total_images

########################## GET_IMAGE_COUNT #############################
# recursivly checks driectory for images
def extract_sub_dirs(image_path, current_path):
    current_path = (current_path + "/").replace("//", "/")
    new_images = False
    for filename in os.listdir(current_path):
        if '.jpg' in filename:
            new_images = True
    if new_images:
        for filename in os.listdir(current_path):
            if os.path.isdir(current_path + filename):
                extract_sub_dirs(image_path, current_path + filename)
            elif '.jpg' in filename or ".xml" in filename:
                shutil.move(current_path + filename, image_path + filename)


###################### EXTRACT IMG FROM XML #############################
def extract_img_from_xml(path):
    if os.path.isdir(path):
        path = (path + "/").replace("//", "/")
        for file in os.listdir(path):
            extract_img_from_xml(path + file)

    elif ".xml" in path:
        tree = ET.parse(path)
        root = tree.getroot()
        for img in root.findall('img'):
            base64_str = img.text.replace('Optional("', "").replace('=")', "")
            base64_remainder = len(base64_str) % 4
            while base64_remainder > 0:
                base64_str += "="
                base64_remainder -= 1
            filename = path.replace(".xml", ".jpg")
            decoded_str = base64.b64decode(base64_str)

            with open(filename, 'wb') as f:
                f.write(decoded_str)


########################## REMOVE_EMPTY_FOLDERS #############################
# recursivly checks driectory for images
def remove_empty_folders(path):
    for file in os.listdir(path):
        if os.path.isdir(path + file):
            file_path = path + file
            if len(os.listdir(file_path)) == 0:
                os.rmdir(file_path)

########################## GET OUTPUT FILE #############################
def get_output_file():
    path = "./logs/"
    file = "workbench_log.txt"
    file_count = 1
    file_path =  path + file
    if not os.path.exists(path):
        os.mkdir(path)
    while os.path.exists(file_path):
        file_path = path + "workbench_log-" + str(file_count) + ".txt"
        file_count += 1
    return file_path

########################## GET_IMAGE_COUNT #############################
# recursivly checks driectory for images
def get_img_count(path):
    # Method Variables

    total_images = 0
    # For every file image in the image dir, check if it has an xml file and move it
    if os.path.isdir(path):
        path = (path + "/").replace("//", "/")
        for filename in os.listdir(path):
            total_images += get_img_count(path + filename)
    if  '.jpg' in path:
        found_label = False
        xml_version = path.split(".jpg")[0] + ".xml"
        total_images += 1
        # check if xml for image was found
        if os.path.exists(xml_version):
            found_label = True

        # if image was found but label was not:
        if found_label == False:
            unlabelled_files.append(path)
    return total_images


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
        if '.jpg' in filename:
            found_label = False
            xml_version = filename.split(".")[0] + ".xml"
            current_image += 1

            # move to test
            if (current_image % 10 == 0) or (total_images < 10 and current_image == total_images):
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
        if '.png' in filename or '.jpg' in filename or '.jpeg' in filename:
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
        if '.png' in file or '.jpg' in file or '.jpeg' in file:
            file_count += 1
            xml_version = file.split(".")[0] + ".xml"
            if file_count in valid_images:
                shutil.move(train_image_path + file, val_image_path)
                if path.exists(train_image_path + xml_version):
                    shutil.move(train_image_path + xml_version, val_image_path)


########################## GET CLASSIFIERS #############################
# Reads all the xml files and gathers all the unique classifiers
def get_classifiers(data_dir):
    class_counter = 0
    classifiers = []
    if os.path.isdir(data_dir):
        data_dir = (data_dir + "/").replace("//", "/")
        for file in os.listdir(data_dir):
            nested_folder = get_classifiers(data_dir + "/" + file)
            if classifiers == [] or classifiers == None:
                classifiers = nested_folder
            else:
                classifiers = classifiers + nested_folder

    if ".xml" in data_dir:
        tree = ET.parse(data_dir)
        root = tree.getroot()
        for object in root.findall('object'):
            for name in object.findall('name'):
                classifiers.append(name.text)
    return classifiers

######################### PRINT CLASSIFIERS ####################################
def classifier_remove_dup(classifiers):
    print("\n\tClassifiers:")
    total_classifiers = 0
    new_list = []
    classes = 0
    while classes < len(classifiers):
        if classifiers[classes] not in new_list:
            new_list.append(classifiers[classes])
        classes += 1
    new_list.sort()
    classes = 0
    while classes < len(new_list):
        print("\t\t" + str(new_list[classes]) + ": " + str(classifiers.count(new_list[classes])))
        total_classifiers += classifiers.count(new_list[classes])
        classes += 1
    print("\n\tTotal Objects: " + str(total_classifiers))
    return new_list

########################## CREATE_CLASSIFIER_NAMES #############################
# takes in a list of all classifiers and writes to the CLASSIFIER_FILE each classifier
def create_classifier_file(file, classifiers):
    stored_lines = []
    class_counter = 0
    with open(file, "w") as f:
        for classification in classifiers:
            class_counter += 1
            f.write(classification + "\n")


########################## GET_LAST_CHECKPOINT #############################
# gets the name of the last classifier from training
def get_last_checkpoint(checkpoint_path):
    last_checkpoint_num = -1
    last_checkpoint = ""
    if checkpoint_path[:-1] != "/":
        checkpoint_path += "/"
    if os.path.exists(checkpoint_path):
        for filename in os.listdir(checkpoint_path):
            if os.path.isdir(filename):
                temp_check = get_last_checkpoint(filename)
                if  last_checkpoint_num < get_checkpoint_int(temp_check):
                    last_checkpoint = checkpoint_path + temp_check
            if CHECKPOINT_KEYWORD and '.tf.index' in filename:
               if last_checkpoint_num < get_checkpoint_int(filename):
                   last_checkpoint_num = get_checkpoint_int(filename)
                   last_checkpoint = checkpoint_path + filename
    if last_checkpoint_num == -1:
        last_checkpoint = ERROR
    return last_checkpoint


########################## GET INPUT ###########################
def get_input(input, split_char):
    if split_char in input:
        input = input.split(split_char)[1]
        input = input.strip()
    return input


########################## GET NUM CLASSES ##########################
def get_num_classes(file):
    num_classes = 0
    if not os.path.exists(file):
        open(file, "w")
    with open(file, "r") as f:
        for line in f.readlines():
            if len(line.strip()) != 0 :
                num_classes = num_classes + 1
    return num_classes


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
        if checkpoint_output == default_output:
            for file in os.listdir(checkpoint_output):
                shutil.move(checkpoint_output + file, current_sess)


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


###### Split path
# Takes a path and returns everying after the workbench directory
def from_workbench(path):
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


############################ GET CHECKPONT EXTENSION ###############################
# Example: yolov3_train_1.tf.index
# returns: .tf.index
def rename_checkpoints(checkpoint_path, max_checkpoints):
    checkpoint_count = get_checkpoint_count(checkpoint_path)
    if checkpoint_count != max_checkpoints:
        return

    files_renamed = 0
    oldest_file = get_oldest_checkpoint(checkpoint_path)
    oldest_file_int = get_checkpoint_int(oldest_file)
    oldest_file_diff = max_checkpoints - oldest_file_int
    while oldest_file_diff != 0:
        oldest_file_int += 1
        for file in os.listdir(checkpoint_path):
            if CHECKPOINT_KEYWORD in file:
                if get_checkpoint_name(file) + str(oldest_file_int) in file:
                    old_file = checkpoint_path + file
                    new_file = checkpoint_path + get_checkpoint_name(file) + str(files_renamed) + file.split("_" + str(get_checkpoint_int(file)))[1]
                    os.rename(old_file, new_file)
        files_renamed -= 1
        oldest_file_diff -= 1
    current_checkpoint = max_checkpoints + files_renamed
    set_to_value = max_checkpoints

    while set_to_value > 0:
        for file in os.listdir(checkpoint_path):
            if CHECKPOINT_KEYWORD + str(current_checkpoint) in file:
                old_file = checkpoint_path + file
                new_file = checkpoint_path + get_checkpoint_name(file) + str(set_to_value) + file.split("_" + str(current_checkpoint))[1]
                os.rename(old_file, new_file)
        set_to_value -= 1
        current_checkpoint -= 1


##################### WRITE TO CHECKPOINT FILE ########################
def write_to_checkpoint(checkpoint_name, filename):
    quote = '"'
    if CHECKPOINT_KEYWORD in checkpoint_name:
        checkpoint_name = CHECKPOINT_KEYWORD + checkpoint_name.split(CHECKPOINT_KEYWORD)[1]
        models = "model_checkpoint_path: "
        all_models = "all_model_checkpoint_paths: "
        with open(filename, "w") as f:
            f.write(models + quote + checkpoint_name + quote)
            f.write("\n")
            f.write(all_models + quote + checkpoint_name + quote)


############################ GET CHECKPOINT NAME ###############################
# Example: yolov3_train_1.tf.index
# returns: yolov3_train_
def get_checkpoint_name(file):
    try:
        file_name = file.split(".")[0].split("_")
        file_name = file_name[0] + "_" + file_name[1] + "_"
        return file_name
    except:
        return file


############################ GET CHECKPOINT INT ###############################
# Example: yolov3_train_1.tf.index
# returns: 1
def get_checkpoint_int(file):
    try:
        file_name = file.split(".")[0]
        return int(file_name.split(CHECKPOINT_KEYWORD)[1])
    except:
        return 0


############################ GET OLDEST CHECKPOINT ###############################
# return the checkpoint with the oldest "last_modified" value
def get_oldest_checkpoint(checkpoint_path):
    oldest = -1;
    checkpoint = checkpoint_path
    oldest_file = ""
    for file in os.listdir(checkpoint_path):
        if CHECKPOINT_KEYWORD in file:
            last_modified = time.ctime(os.path.getmtime(checkpoint + file)).split(' ')
            last_modified = int(str(last_modified[4]) + str(get_month(last_modified[1]))  +  str(last_modified[3]).replace(":", ""))
            if last_modified > oldest:
                oldest_file = file
                oldest = last_modified
    return oldest_file


##################### GET CHECKPOINT COUNT ###############################
def get_checkpoint_count(path):
    chekpoint_count = 0
    for files in os.listdir(path):
        if CHECKPOINT_KEYWORD in files and ".tf.index" in files:
            chekpoint_count += 1
    return chekpoint_count


############################ GET MONTH ###############################
# takes three letter abreviation of month and returns it month number
# if invalid string, returns 0
def get_month( month ):
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

############################ DUPLICATE PB ###########################
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

def remove_temp(new_path, temp_path):
    new_path = (new_path + "/").replace("//", "/")
    temp_path = (temp_path + "/").replace("//", "/")
    shutil.rmtree(new_path + "assets")
    shutil.rmtree(new_path + "variables")
    for file in os.listdir(temp_path):
        if ".pb" not in file:
            if os.path.exists(new_path + file):
                os.remove(new_path + file)
            shutil.move(temp_path + file, new_path + file)
    shutil.rmtree(temp_path)


############################ FILE IS VALID ###########################
# Returns true if the file exists and is not empty, else false
def is_valid(file):
    return os.path.exists(file) and os.stat(file).st_size != 0
