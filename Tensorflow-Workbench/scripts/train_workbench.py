from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import math
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset



def run_train(train_dataset_in, val_dataset_in, tiny, images,
              weights, classifiers, mode, transfer, size, epochs, batch_size,
              learning_rate, num_classes, weights_num_classes, checkpoint_path, total_checkpoints):

    checkpoint_path = checkpoint_path.replace("\\", "/")

    if tiny:
        model = YoloV3Tiny(size=size, training=True,
                           classes=num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(size=size, training=True, classes=num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks
    try:
        train_dataset = dataset.load_fake_dataset(images)
    except:
        print("\n\tERROR: training needs an img in the data folder or your input image folder" )
        return False

    if train_dataset_in:
        train_dataset = dataset.load_tfrecord_dataset(
            train_dataset_in, classifiers, size)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, size),
        dataset.transform_targets(y, anchors, anchor_masks, size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_fake_dataset(images)
    if val_dataset_in:
        val_dataset = dataset.load_tfrecord_dataset(
            val_dataset_in, classifiers, size)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, size),
        dataset.transform_targets(y, anchors, anchor_masks, size)))

    # Configure the model for transfer learning
    if transfer == 'none':
        pass  # Nothing to do
    elif transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes

        # reset top layers
        if tiny:
            model_pretrained = YoloV3Tiny(
                size=size, training=True, classes=weights_num_classes or num_classes)
        else:
            model_pretrained = YoloV3(
                size=size, training=True, classes=weights_num_classes or num_classes)
        model_pretrained.load_weights(weights)

        if transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))

        elif transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                    freeze_all(l)

    else:
        # All other transfer require matching classes
        model.load_weights(weights)
        if transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif transfer == 'frozen':
            # freeze everything
            freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss = [YoloLoss(anchors[mask], classes=num_classes)
            for mask in anchor_masks]

    if mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        print("\nEager Training Starting...\n")
        print("\n\t NOTICE: This mode is mainly for debugging and should not be used for real training\n")
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info("{}_val_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(
                checkpoint_path +'yolov3_train_{}.tf'.format(epoch))
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(mode == 'eager_fit'))
        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, monitor='val_loss', verbose=1),
            ModelCheckpoint(checkpoint_path + 'yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]
        total_runs = math.floor(epochs/total_checkpoints)
        runs_remainder = epochs % total_checkpoints
        run = 1
        extra_run = 0
        print("\n===================================================================================================\n")
        if runs_remainder != 0:
            extra_run = 1

        if total_checkpoints > 0 and total_checkpoints < epochs:
            print("\tTraining in runs to save memory")
            while run <= total_runs:
                print("\n=======================================")
                print("             Training Run " + str(run) + "/" + str(total_runs + extra_run))
                print("=======================================\n")
                history = model.fit(train_dataset,
                                    epochs=total_checkpoints,
                                    callbacks=callbacks,
                                    validation_data=val_dataset)

                # Increment the run
                run += 1

            if runs_remainder != 0:
                print("\n=======================================")
                print("             Training Run " + str(run) + "/" + str(total_runs + extra_run))
                print("=======================================\n")
                history = model.fit(train_dataset,
                                     epochs=runs_remainder,
                                     callbacks=callbacks,
                                     validation_data=val_dataset)
                # plot training history
                # pyplot.plot(history.history['loss'], label='train')
                # pyplot.plot(history.history['val_loss'], label='test')
                # pyplot.legend()
                # pyplot.show()

        else:
            history = model.fit(train_dataset,
                                epochs=epochs,
                                callbacks=callbacks,
                                validation_data=val_dataset)

    return True





def find_lowest_check(checkpoints):
    lowest_check = checkpoints[0]
    for checks in checkpoints:
        if checks < lowest_check:
            lowest_check = checks
    return lowest_check

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
