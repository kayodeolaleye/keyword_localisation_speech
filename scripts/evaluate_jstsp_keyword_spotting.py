import pdb
import sys

import numpy as np

from sklearn.metrics import average_precision_score

from scripts.show_results_plos_one import load_data
sys.path.insert(0, "Interspeech")
from train_emb import Flickr8kDataset, LabelsTextLoader


def eval_keyword_spotting(data):
    samples = Flickr8kDataset.load_samples("test")
    labels_loader = LabelsTextLoader()
    vocab = labels_loader.vocab

    # key_to_scores = {sample["key"]: sample["utt-score"] for sample in data}
    scores = np.vstack([data[s.value] for s in samples])
    labels = np.vstack([labels_loader(s) for s in samples])

    ap = 100 * average_precision_score(labels, scores)
    word_ap = [
        (word, 100 * average_precision_score(labels[:, i], scores[:, i]))
        for word, i in vocab.items()
    ]

    return ap, word_ap


data = np.load("Interspeech/output/outputs_204060_ws_untrimmed/all_full_sigmoid_out.npz")
# data, _ = load_data(to_trim=False)
mean_ap, word_ap = eval_keyword_spotting(data)
for word, ap in word_ap:
    print("{:10s} {:5.2f}%".format(word, ap))
print("{:10s} {:5.2f}%".format("mean", mean_ap))
print()
