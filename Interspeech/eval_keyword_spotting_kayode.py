import click
import pdb

import numpy as np

from sklearn.metrics import average_precision_score

from train_emb import (
    Flickr8kYorubaDataset,
    LabelsTextLoader,
)

from predict_emb import VOCAB_CHOICES



def main():
    data = np.load("output/yoruba-results-1.npz")
    samples = Flickr8kYorubaDataset.load_samples("lagos", "test")
    samples = [s for s in samples if s.value in data.keys()]

    labels_loader = LabelsTextLoader()
    vocab = labels_loader.vocab

    num_samples = len(samples)
    num_words = len(vocab)

    scores = np.vstack([data[sample.value] for sample in samples])
    labels = np.vstack([labels_loader(s) for s in samples])

    mean_ap = 100 * average_precision_score(labels, scores)
    word_ap = [
        (word, 100 * average_precision_score(labels[:, i], scores[:, i]))
        for word, i in vocab.items()
    ]

    for word, ap in word_ap:
        print("{:10s} {:5.2f}%".format(word, ap))
    print("{:10s} {:5.2f}%".format("mean", mean_ap))
    print()


if __name__ == "__main__":
    main()
