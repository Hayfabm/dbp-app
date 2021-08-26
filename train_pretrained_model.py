import datetime
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)


from utils import create_dataset, one_hot_encoding


if __name__ == "__main__":

    # fix the seed
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # training parameters
    BATCH_SIZE = 128
    NUM_EPOCHS = 2000

    # create train dataset
    sequences_train, labels_train = create_dataset(data_path="data/PDB14189.csv")

    # create test dataset
    sequences_test, labels_test = create_dataset(data_path="data/PDB2272.csv")

    # encode sequences
    sequences_train_encoded = np.concatenate(
        [one_hot_encoding(seq) for seq in sequences_train], axis=0
    )  # (14189, 800, 20)
    sequences_test_encoded = np.concatenate(
        [one_hot_encoding(seq) for seq in sequences_test], axis=0
    )  # (2272, 800, 20)

    # encode labels
    labels_train_encoded = to_categorical(
        labels_train, num_classes=2, dtype="float32"
    )  # (14189, 2)
    labels_test_encoded = to_categorical(
        labels_test, num_classes=2, dtype="float32"
    )  # (2272, 2)

    # load model
    model = load_model("checkpoint/model_0.hdf5")
    print(model.summary())

    # compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # tf.config.experimental_run_functions_eagerly(True)

    # in order to see logs, please run this command: tensorboard --logdir logs/
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    my_callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1),
        EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1),
        ModelCheckpoint(
            filepath=log_dir + "/model.{epoch:02d}-{val_accuracy:.2f}.hdf5"
        ),
        TensorBoard(
            log_dir=log_dir, update_freq="epoch", profile_batch=0, histogram_freq=1
        ),
    ]

    # fit the model
    history = model.fit(
        sequences_train_encoded,
        labels_train_encoded,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
        validation_data=(sequences_test_encoded, labels_test_encoded),
        callbacks=my_callbacks,
    )
