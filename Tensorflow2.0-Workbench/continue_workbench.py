import os
from os import path
sys.path.insert(1, './scripts')

import defaults
import files
import preferences


############################## MAIN ##########################
def main():
    # get help if needed
    defaults.get_help()

    # check if file paths exists
    files.checkIfNecessaryPathsAndFilesExist()

    # Display pref
    print("\nCurrent Preferences:")
    preferences.print_pref()

    # generating tensorflow models
    print("\nGenerating TensorFlow model...")
    chkpnt_weights = files.get_last_checkpoint()
    if path.isfile(preferences.validate_input):
        create_tf_model.run_export_tfserving(chkpnt_weights,
                                                  preferences.tiny,
                                                  preferences.output_model,
                                                  preferences.classifier_file,
                                                  preferences.validate_input + file,
                                                  preferences.num_classes)
    else:
        for file in os.listdir(preferences.validate_input):
            if '.jpg' in file:
                create_tf_model.run_export_tfserving(chkpnt_weights,
                                                          preferences.tiny,
                                                          preferences.output_model,
                                                          preferences.classifier_file,
                                                          preferences.validate_input + file,
                                                          preferences.num_classes)
    print("\n\tTensorFlow model Generated!")

main()
