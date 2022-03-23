"""Random performance for keyword spotting."""

import click
import numpy as np

from sklearn.metrics import average_precision_score

from train_emb import (
    Flickr8kDataset,
    LabelsTextLoader,
)

from predict_emb import VOCAB_CHOICES


@click.command()
@click.option(
    "--vocab",
    "vocab_type",
    type=click.Choice(VOCAB_CHOICES),
    help="keywords to compute the metrics on",
)
def main(vocab_type):
    samples = Flickr8kDataset.load_samples("test")
    labels_loader = LabelsTextLoader(vocab_type)
    vocab = labels_loader.vocab

    num_samples = len(samples)
    num_words = len(vocab)

    scores = np.random.rand(num_samples, num_words)
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
