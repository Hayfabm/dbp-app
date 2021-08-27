"""utils file"""
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def create_dataset(data_path: str) -> Tuple[List[str], List[int]]:
    dataset = pd.read_csv(data_path)
    dataset = dataset.sample(frac=1).reset_index(drop=True)  # shuffle the dataset
    return list(dataset["sequence"]), list(dataset["label"])


def split_dataset(
    sequences_list: List[str], labels_list: List[int], train_size: float = 0.8
) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    dataset = pd.DataFrame({"sequence": sequences_list, "label": labels_list})
    dataset = dataset.sample(frac=1, random_state=1)
    train, remaining = train_test_split(dataset, train_size=train_size, random_state=2)
    valid, test = train_test_split(remaining, test_size=0.5, random_state=3)
    x_train, x_valid, x_test = train["sequence"], valid["sequence"], test["sequence"]
    y_train, y_valid, y_test = train["label"], valid["label"], test["label"]
    return (
        list(x_train),
        list(x_valid),
        list(x_test),
        list(y_train),
        list(y_valid),
        list(y_test),
    )


def one_hot_encoding(
    sequence: str,
    max_seq_length: int = 800,
    CONSIDERED_AA: str = "ACDEFGHIKLMNPQRSTVWY",
):
    # adapt sequence size
    if len(sequence) > max_seq_length:
        # short the sequence
        sequence = sequence[:max_seq_length]
    else:
        # pad the sequence
        sequence = sequence + "." * (max_seq_length - len(sequence))

    # encode sequence
    encoded_sequence = np.zeros((max_seq_length, len(CONSIDERED_AA)))  # (800, 20)
    for i, amino_acid in enumerate(sequence):
        if amino_acid in CONSIDERED_AA:
            encoded_sequence[i][CONSIDERED_AA.index(amino_acid)] = 1
    model_input = np.expand_dims(encoded_sequence, 0)  # add batch dimension

    return model_input  # (1, 800, 20)


def preprocess_word_embedding_encoding(
    sequence: str,
    max_seq_length: int = 800,
    CONSIDERED_AA: str = "ACDEFGHIKLMNPQRSTVWY",
):
    # amino acids encoding
    aa_mapping = {aa: i + 1 for i, aa in enumerate(CONSIDERED_AA)}

    # adapt sequence size
    if len(sequence) > max_seq_length:
        # short the sequence
        sequence = sequence[:max_seq_length]
    else:
        # pad the sequence
        sequence = sequence + "." * (max_seq_length - len(sequence))

    # encode sequence
    encoded_sequence = np.zeros((max_seq_length,))  # (800,)
    for i, amino_acid in enumerate(sequence):
        if amino_acid in CONSIDERED_AA:
            encoded_sequence[i] = aa_mapping[amino_acid]
    model_input = np.expand_dims(encoded_sequence, 0)  # add batch dimension

    return model_input  # (1, 800)

