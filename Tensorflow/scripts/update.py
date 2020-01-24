import csv

def main():
    LABEL_MAP = "../data/label_map.pbtxt"
    TEST_FILE =  "test.py"
    GENERATE_REC = "generate_tfrecords.py"
    PIPELINE = "../training/pre-trained_model/pipeline.config"
    TRAIN_CSV_FILE = '../data/train_labels.csv'

    class_counter = 0
    classes = []

    with open(TRAIN_CSV_FILE) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[3] not in classes:
                if row[3] != "class":
                    classes.append(row[3])

    NUM_CLASSES = str(len(classes)) + "\n"

########################## TEST_FILE #############################
    stored_lines = []
    with open(TEST_FILE, "r") as f:
        for line in f.readlines():
            if 'NUM_CLASSES =' in line:
                stored_lines.append("NUM_CLASSES = " + NUM_CLASSES)
            else:
                stored_lines.append(line)

    with open(TEST_FILE, "w") as f:
        for line in stored_lines:
            f.write(line)

########################## PIPELINE #############################
    stored_lines = []
    with open(PIPELINE, "r") as f:
        for line in f.readlines():
            if 'num_classes:' in line:
                stored_lines.append("\tnum_classes: " + NUM_CLASSES)
            else:
                stored_lines.append(line)

    with open(PIPELINE, "w") as f:
        for line in stored_lines:
            f.write(line)

########################## LABEL_MAP ############################
    stored_lines = []
    with open(LABEL_MAP, "w") as f:
        for classifier in classes:
            class_counter += 1
            f.write("item { \n\t\tid: " + str(class_counter) + "\n\t\tname: '" + classifier + "'\n}\n\n")

########################## GENERATE_REC ############################
    stored_lines = []
    class_counter = 1
    change = False
    with open(GENERATE_REC, "r") as f:
        for line in f.readlines():
            if 'def classAsTextToClassAsInt' in line:
                stored_lines.append("def classAsTextToClassAsInt(classAsText):\n\n")
                for classifier in classes:
                    if class_counter == 1:
                        stored_lines.append("\tif classAsText == '" + classifier + "':\n\t\t return " + str(class_counter) + "\n")
                    else:
                        stored_lines.append("\telif classAsText == '" + classifier + "':\n\t\t return " + str(class_counter) + "\n")
                    class_counter += 1
                change = True
                stored_lines.append("\telse: \n\t\treturn -1 \n\n\nif __name__ == '__main__':\n\tmain()")
            elif not change:
                stored_lines.append(line)

    with open(GENERATE_REC, "w") as f:
        for line in stored_lines:
            f.write(line)

if __name__ == "__main__":
    main()
