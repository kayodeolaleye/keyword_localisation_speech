import os
import pdb
import pickle

import click
import numpy as np
import streamlit as st

from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from toolz import partition

from scripts.evaluate_visual_vs_text import (
    VISUAL_MODELS,
    BASE_PATH,
    config,
    load_true,
    wav_to_img_path,
)


def load(path, parser):
    with open(path, "r") as f:
        return list(map(parser, f.readlines()))


def parse_token(line):
    key, *words = line.strip().split()
    text = " ".join(words)
    img, i = key.split("#")
    key1 = img.split(".")[0] + "_" + str(i)
    return key1, text


def wav_path_to_key(wav_path):
    _, filename = os.path.split(wav_path)
    key, _ = os.path.splitext(filename)
    return key


def show1(data, true, pred, word):
    word_id = data["VOCAB"][word]
    true1 = true[:, word_id]
    pred1 = pred[:, word_id]

    num_samples = len(true1)

    # quantitative metrics
    precision, recall, _ = precision_recall_curve(true1, pred1)
    fpr, tpr, _ = roc_curve(true1, pred1)

    fig, axs = plt.subplots(ncols=2, figsize=(5.5, 2.5), tight_layout=True)

    axs[0].plot(recall, precision)
    axs[0].set_xlabel("recall")
    axs[0].set_ylabel("precision")
    axs[0].set_title("AUPR: {:.1f}%".format(100 * auc(recall, precision)))

    axs[1].plot(fpr, tpr)
    axs[1].set_xlabel("FPR")
    axs[1].set_ylabel("TPR")
    axs[1].set_title("AUROC: {:.1f}%".format(100 * auc(fpr, tpr)))

    col1, _ = st.beta_columns(2)
    col1.markdown("## Quantitative results")
    col1.pyplot(fig)

    def prepare_data(sample_id, rank):
        sample = data["test"][sample_id]
        wav_path = sample["wave"]
        img_path = wav_to_img_path(wav_path)

        key = wav_path_to_key(wav_path)
        transcript = data["key_to_transcript"][key]
        words_sample = [data["vocab_inv"][w] for w in np.where(true[sample_id])[0]]
        description = "\n".join(
            [
                "key:        " + key,
                "rank:       " + str(rank),
                "is-correct: " + str(bool(true1[sample_id])),
                "score:      " + "{:.2f}".format(pred1[sample_id]),
                "labels:     " + ", ".join(words_sample),
                "transcript: " + transcript,
            ]
        )

        return img_path, description

    st.markdown("## Top 10 predictions")
    ranks = np.arange(num_samples)
    rank_to_sample_id = pred1.argsort()[::-1]
    samples_top10 = rank_to_sample_id[:10]
    data_top10 = [
        prepare_data(sample_id, rank + 1)
        for rank, sample_id in zip(ranks[:10], samples_top10)
    ]
    for group in partition(5, data_top10):
        columns = st.beta_columns(5)
        for col, (img_path, description) in zip(columns, group):
            col.image(img_path)
            col.code(description)

    st.markdown("## Bottom 5 positives")
    rank_to_true = true1[rank_to_sample_id]
    rank_eq_true = np.where(rank_to_true)[0]
    rank_eq_true_5 = rank_eq_true[-5:]
    samples_bot5pos = rank_to_sample_id[rank_eq_true_5]
    data_bot5 = [
        prepare_data(sample_id, rank + 1)
        for rank, sample_id in zip(ranks[rank_eq_true_5], samples_bot5pos)
    ]
    columns = st.beta_columns(5)
    for col, (img_path, description) in zip(columns, data_bot5):
        col.image(img_path)
        col.code(description)


def main():
    with open(os.path.join(BASE_PATH, config.pickle_file), "rb") as f:
        data = pickle.load(f)

    file_transcript = os.path.join(config.flickr8k_trans_dir, "Flickr8k.token.txt")
    key_to_transcript = dict(load(file_transcript, parse_token))
    data["key_to_transcript"] = key_to_transcript

    vocab = data["VOCAB"]
    vocab_inv = {i: w for w, i in vocab.items()}
    words = [vocab_inv[i] for i in range(len(vocab))]
    data["vocab_inv"] = vocab_inv

    st.set_page_config(layout="wide")
    st.title("Visual predictions versus textual labels on Flickr8K")

    col1, _ = st.beta_columns(2)

    col1.markdown("## Options")
    visual_model = col1.selectbox("visual model", VISUAL_MODELS)

    true = load_true(data)
    pred = VISUAL_MODELS[visual_model](data)

    for word in words:
        st.markdown("# " + word)
        show1(data, true, pred, word)
        st.markdown("---")


if __name__ == "__main__":
    main()
