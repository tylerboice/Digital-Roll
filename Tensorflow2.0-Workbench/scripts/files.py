import os
import random
import shutil

from os import path
unlabelled_files = []

########################## Checking FOR FILES #############################
# checks if all necessary files exist
def checkIfNecessaryPathsAndFilesExist(image_path, min_images, output_path, test_image_path,
                                       train_image_path, val_image_path, weights_path):

    ####### IMAGE PATH #######
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    images_found = check_for_images(image_path, min_images)

    if images_found:
        ####### OUTPUT MODEL PATH #######
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
    print("\nTotal images = " + str(total_images))
    print("Images not labelled = " + str(len(unlabelled_files)))
    if len(unlabelled_files) != 0:
        print("\nThe following images do not have an xml file:")
        for item in unlabelled_files:
            print("\t" + item)

    # No images Found
    if total_images == 0:
        print("\nERROR: No images have been found in the image folder")
        print("\n\tImage Folder Location: " + path)
        print("\n\tFor an example set, look at the Pre_Labeled_Images folder in the repository or at https://github.com/tylerboice/Digital-Roll")
        return False

    # total images must be greater than pre-defined count to train on
    elif total_images < min_images:
        print("\n\nTensorflow needs at least " + str(min_images) + " images to train")
        return False

    return True

########################## GET_IMAGE_COUNT #############################
# recursivly checks driectory for images
def get_img_count(path):
    # Method Variables
    total_images = 0

    # For every file image in the image dir, check if it has an xml file and move it
    for filename in os.listdir(path):
        if os.path.isdir(path + filename):
            total_images += get_img_count(path + filename)
        elif '.png' in filename or '.jpg' in filename or '.jpeg' in filename:
            found_label = False
            xml_version = "/" +filename.split(".")[0] + ".xml"
            total_images += 1
            # check if cml for image was found
            if os.path.exists(path + xml_version):
                found_label = True

            # if image was found but label was not:
            if found_label == False:
                unlabelled_files.append(filename)
    return total_images

########################## SORT IMAGES #############################
# Takes all the images in the image folder and places them in test, train and validate
# Train = 90% of the images
# Test = 10% of the images
# Validate = takes user spicifed amount of files out of train (num_validate)
def sort_images(num_validate, image_path, test_image_path, train_image_path, val_image_path):

    # Method Variables
    total_images = 0
    train_images = 0
    valid_images = []
    get_valid = False

    # For every file image in the image dir, check if it has an xml file and move it
    for filename in os.listdir(image_path):
        if '.png' in filename or '.jpg' in filename or '.jpeg' in filename:
            found_label = False
            xml_version = filename.split(".")[0] + ".xml"
            total_images += 1

            # move to test
            if total_images % 10 == 0:
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

    # count all image and .xml files in test
    for filename in os.listdir(test_image_path):
        if '.png' in filename or '.jpg' in filename or '.jpeg' in filename:
            total_images += 1

    # count all image and .xml files in train
    for filename in os.listdir(train_image_path):
        if '.png' in filename or '.jpg' in filename or '.jpeg' in filename:
            total_images += 1
            train_images += 1

    # move all files in validate to train
    for file in os.listdir(val_image_path):
        shutil.move(val_image_path + file, train_image_path)

    # gather all valid images from train
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
    name_tag = "<name>"
    name_end_tag = "</name>"
    for file in os.listdir(data_dir):
        if path.isfile(data_dir + file) == False:
            dir = data_dir + file + "/"
            nested_folder = get_classifiers(dir)
            if classifiers == None or classifiers == []:
                classifiers = nested_folder
            else:
                classifiers = classifiers + list(set(nested_folder) - set(classifiers))

        if ".xml" in file:
            with open(data_dir + file, "r") as f:
                for line in f.readlines():
                    if name_tag in line:
                        name = line.replace(name_end_tag, "")
                        name = name.replace(name_tag, "")
                        name = name.replace("\t", "")
                        name = name.replace("\n", "")
                        if name not in classifiers:
                            classifiers.append(name)
                            classifiers.sort()
    return classifiers



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
    last_checkpoint_num = 0
    last_checkpoint = ""
    if os.path.exists(checkpoint_path):
        for filename in os.listdir(checkpoint_path):
            if 'tf.index' and 'train' in filename:
               if 'of' not in filename:
                   current = filename.split(".")[0]
                   current = current.split('_')[2]
                   if last_checkpoint_num < int(current):
                       last_checkpoint_num = int(current)
                       last_checkpoint = checkpoint_path + filename.split(".")[0] + ".tf"
    if last_checkpoint_num == 0:
        last_checkpoint = checkpoint_path + "yolov3_train_0.tf"
    return last_checkpoint

########################## GET INPUT ###########################
def get_input(input, split_char):
    input = line.split(split_char)[1]
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
def save_checkpoints(checkpoint_output, save_sess_path, max_saved_sess):
    keyword = "saved_session-"
    checkpoints = False
    if not os.path.exists(save_sess_path):
        os.mkdir(save_sess_path)
    current_sess = remove_smallest_sess(save_sess_path, max_saved_sess, keyword) + 1
    current_sess = (save_sess_path + "/" + keyword + str(current_sess)).replace("\\", "/")
    current_sess = current_sess.replace("//", "/")
    for file in os.listdir(checkpoint_output):
        if "train_" in file:
            checkpoints = True
    if checkpoints:
        os.mkdir(current_sess)
        print("\tPrevious session stored in " + split_path(current_sess))
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
        print("\tOldest session: " + split_path(oldest_sess) + " was removed")

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
