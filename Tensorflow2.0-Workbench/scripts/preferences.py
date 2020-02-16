from scripts import defaults
from scripts import files

from absl import app, flags, logging
from absl.flags import FLAGS
import sys

batch_size = defaults.check_preferences(defaults.BATCH_SIZE_VAR, defaults.FLAGS.batch_size, defaults.INT)
checkpoint_output = defaults.check_preferences(defaults.CHECKPOINT_VAR, defaults.FLAGS.checkpoint_path, defaults.FILE)
classifier_file = defaults.check_preferences(defaults.CLASSIFIERS_VAR, defaults.FLAGS.classifiers, defaults.FILE)
dataset_test = defaults.check_preferences(defaults.DATASET_TEST_VAR, defaults.FLAGS.dataset_test, defaults.FILE)
dataset_train = defaults.check_preferences(defaults.DATASET_TRAIN_VAR, defaults.FLAGS.dataset_train, defaults.FILE)
epochs = defaults.check_preferences(defaults.EPOCH_NUM_VAR, defaults.FLAGS.epochs, defaults.INT)
image_size = defaults.check_preferences(defaults.IMAGE_SIZE_VAR, defaults.FLAGS.image_size, defaults.INT)
learning_rate = defaults.check_preferences(defaults.LEARN_RATE_VAR, defaults.FLAGS.learn_rate, defaults.FLOAT)
mode = defaults.check_preferences(defaults.MODE_VAR, defaults.FLAGS.mode, defaults.MODE_OPTIONS)
num_classes = files.get_num_classes(classifier_file)
output =  defaults.check_preferences(defaults.OUTPUT_VAR, defaults.FLAGS.output, defaults.FILE)
pref_file = defaults.FLAGS.pref
tiny = defaults.check_preferences(defaults.TINY_WEIGHTS_VAR, defaults.FLAGS.tiny_weights, defaults.BOOL)
transfer = defaults.check_preferences(defaults.TRANSFER_VAR, defaults.FLAGS.transfer, defaults.TRANSFER_OPTIONS)
validate_input = defaults.check_preferences(defaults.VALID_IN_VAR, defaults.FLAGS.validate_image_input, defaults.FILE)
weight_num_classes = defaults.check_preferences(defaults.WEIGHTS_CLASS_VAR, defaults.FLAGS.weights_path, defaults.INT)


# validate image input
flags.DEFINE_string('weights', defaults.get_weights_path(tiny), 'path to weights')
FLAGS(sys.argv)

weights = FLAGS.weights

def print_pref():
    print("\tBatch Size............. " + str(batch_size))
    print("\tCheckpoint Output...... " + checkpoint_output)
    print("\tClassifier file........ " + classifier_file)
    print("\tDataSet-test........... " + dataset_test)
    print("\tDataSet-train.......... " + dataset_train)
    print("\tEpochs................. " + str(epochs))
    print("\tImage Size............. " + str(image_size))
    print("\tLearning Rate.......... " + str(learning_rate))
    print("\tMode................... " + mode)
    print("\tNumber of Classes...... " + str(num_classes))
    print("\tOutput Model........... " + output)
    print("\tPreference File........ " + pref_file)
    print("\tTiny Weights........... " + str(tiny))
    print("\tTransfer............... " + transfer)
    print("\tValidate Image Input... " + validate_input)
    print("\tWeighted Classes....... " + str(weight_num_classes))
    print("\tWeights Path........... " + weights)
