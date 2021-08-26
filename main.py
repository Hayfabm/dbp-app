"""
WARNINGS: if you run the app locally and don't have a GPU you should choose device='cpu'
"""

from typing import List

from tensorflow.keras.models import load_model
import numpy as np

from utils import create_dataset, encode_sequence


class DBPApp:
    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self.num_gpus = 0 if device == "cpu" else 1

        self.NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
        self.max_seq_length = 600

        # NOTE: if you have issues at this step, please use h5py 2.10.0
        # by running the following command: pip install h5py==2.10.0
        self.model = load_model("checkpoint/model_0.hdf5")
        print(self.model.summary())

    def compute_scores(self, sequences_list: List[str]) -> List[float]:
        scores_list = []
        for sequence in sequences_list:
            # encode sequence
            sequence_encoded = encode_sequence(sequence)

            # forward pass throught the model
            model_output = self.model.predict(sequence_encoded)[0]

            scores_list.append(model_output[1])  # model_output[1]: probability to bind
        return scores_list


if __name__ == "__main__":
    # sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQ", "KALEE", "LAGYNIVATPRGYVLAGG"]

    # sequences, labels = create_dataset(data_path="data/PDB14189.csv")  # 0.7614
    sequences, labels = create_dataset(data_path="data/PDB2272.csv")  # 0.6558

    app = DBPApp("cpu")

    scores = app.compute_scores(sequences)

    # compute accuracy of the model
    acc = []
    for s, l in zip(scores, labels):
        if ((s >= 0.5) and (l == 1)) or ((s < 0.5) and (l == 0)):
            acc.append(1)
        else:
            acc.append(0)
    print(np.mean(acc))
