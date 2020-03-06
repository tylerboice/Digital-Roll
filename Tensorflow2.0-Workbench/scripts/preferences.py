from scripts import defaults
from scripts import files

from absl import app, flags, logging
from absl.flags import FLAGS
import sys


batch_size = defaults.check_preferences(defaults.BATCH_SIZE_VAR,
                                        defaults.FLAGS.batch_size,
                                        defaults.INT)


classifier_file = defaults.check_preferences(defaults.CLASSIFIERS_VAR,
                                             defaults.FLAGS.classifiers,
                                             defaults.FILE)

dataset_test = defaults.check_preferences(defaults.DATASET_TEST_VAR,
                                          defaults.FLAGS.dataset_test,
                                          defaults.FILE)

dataset_train = defaults.check_preferences(defaults.DATASET_TRAIN_VAR,
                                           defaults.FLAGS.dataset_train,
                                           defaults.FILE)

epochs = defaults.check_preferences(defaults.EPOCH_NUM_VAR,
                                    defaults.FLAGS.epochs,
                                    defaults.INT)

image_size = defaults.check_preferences(defaults.IMAGE_SIZE_VAR,
                                        defaults.FLAGS.image_size,
                                        defaults.INT)

max_checkpoints = defaults.check_preferences(defaults.MAX_CHECK_VAR,
                                             defaults.FLAGS.max_checkpoints,
                                             defaults.INT)

max_saved_sess = defaults.check_preferences(defaults.MAX_SESS_VAR,
                                            defaults.FLAGS.max_sessions,
                                            defaults.INT)


mode = defaults.check_preferences(defaults.MODE_VAR,
                                  defaults.FLAGS.mode,
                                  defaults.MODE_OPTIONS)

num_classes = files.get_num_classes(classifier_file)

output = defaults.check_preferences(defaults.OUTPUT_VAR,
                                    defaults.FLAGS.output,
                                    defaults.FILE)

pref_file = defaults.FLAGS.pref


sessions = defaults.check_preferences(defaults.SAVED_SESS_VAR,
                                      defaults.FLAGS.sessions,
                                      defaults.FILE)

tiny = defaults.check_preferences(defaults.TINY_WEIGHTS_VAR,
                                  defaults.FLAGS.tiny_weights,
                                  defaults.BOOL)

transfer = defaults.check_preferences(defaults.TRANSFER_VAR,
                                      defaults.FLAGS.transfer,
                                      defaults.TRANSFER_OPTIONS)

validate_img_num = defaults.check_preferences(defaults.VALID_IMGS_VAR,
                                              defaults.FLAGS.val_img_num,
                                              defaults.INT)

validate_input = defaults.check_preferences(defaults.VALID_IN_VAR,
                                            defaults.FLAGS.val_image_path,
                                            defaults.FILE)

weight_num_classes = defaults.check_preferences(defaults.WEIGHTS_NUM_VAR,
                                                defaults.FLAGS.weighted_classes,
                                                defaults.INT)


# validate image input
flags.DEFINE_string(defaults.WEIGHTS_PATH_VAR, defaults.get_weights_path(tiny), 'path to weights')
FLAGS(sys.argv)

weights = FLAGS.weights
