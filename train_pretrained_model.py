import tensorflow as tf

from keras.models import load_model

from utils import create_dataset, split_dataset, encode_sequence


if __name__ == "__main__":
    # training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 200

    # create dataset
    sequences, labels = create_dataset(data_path="data/PDB14189.csv")

    # split the dataset
    (
        sequences_train,
        sequences_valid,
        sequences_test,
        labels_train,
        labels_valid,
        labels_test,
    ) = split_dataset(sequences, labels, train_size=0.8)

    # encode sequences
    sequences_train_encoded = tf.keras.layers.Concatenate(axis=0)(
        [encode_sequence(seq) for seq in sequences_train]
    )  # (11351, 600, 20)
    sequences_valid_encoded = tf.keras.layers.Concatenate(axis=0)(
        [encode_sequence(seq) for seq in sequences_valid]
    )  # (1419, 600, 20)
    sequences_test_encoded = tf.keras.layers.Concatenate(axis=0)(
        [encode_sequence(seq) for seq in sequences_test]
    )  # (1419, 600, 20)

    # encode labels
    labels_train_encoded = tf.keras.utils.to_categorical(
        labels_train, num_classes=2, dtype="float32"
    )  # (11351, 2)
    labels_valid_encoded = tf.keras.utils.to_categorical(
        labels_valid, num_classes=2, dtype="float32"
    )  # (1419, 2)
    labels_test_encoded = tf.keras.utils.to_categorical(
        labels_test, num_classes=2, dtype="float32"
    )  # (1419, 2)

    # load model
    model = load_model("checkpoint/model_0.hdf5")
    print(model.summary())

    # compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    tf.config.experimental_run_functions_eagerly(True)

    # fit the model
    history = model.fit(
        sequences_train_encoded,
        labels_train_encoded,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
        validation_data=(sequences_valid_encoded, labels_valid_encoded),
    )
