import os
import pdb
import pickle

from typing import Any, Dict, List

import click
import numpy as np
import streamlit as st

from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from toolz import partition

from scripts.data import (
    BASE_PATH,
    config,
    get_key_img,
    load,
    parse_token,
    wav_path_to_key,
    wav_to_img_path,
)
from scripts.evaluate import (
    MODELS,
    load_true,
)
from scripts.evaluate_group_by_image import group_true_and_pred


def reverse_non_injective_dict(d):
    q = {}  # type: Dict[Any, List[Any]]
    for k, v in d.items():
        q.setdefault(v, []).append(k)
    return q


def show1(true, pred, vocab, word, prepare_sample):
    word_id = vocab[word]
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

    st.markdown("## Top 10 predictions")
    ranks = np.arange(num_samples)
    rank_to_sample_id = pred1.argsort()[::-1]
    samples_top10 = rank_to_sample_id[:10]
    data_top10 = [
        prepare_sample(word_id, sample_id, rank + 1)
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
        prepare_sample(word_id, sample_id, rank + 1)
        for rank, sample_id in zip(ranks[rank_eq_true_5], samples_bot5pos)
    ]
    columns = st.beta_columns(5)
    for col, (img_path, description) in zip(columns, data_bot5):
        col.image(img_path)
        col.code(description)


def main():
    TO_SHOW_SINGLE_WORD = True

    with open(os.path.join(BASE_PATH, config.pickle_file), "rb") as f:
        data = pickle.load(f)

    file_transcript = os.path.join(config.flickr8k_trans_dir, "Flickr8k.token.txt")
    key_to_transcript = dict(load(file_transcript, parse_token))

    vocab = data["VOCAB"]
    samples = data["test"]
    keys = [wav_path_to_key(sample["wave"]) for sample in samples]

    vocab_inv = {i: w for w, i in vocab.items()}
    words = [vocab_inv[i] for i in range(len(vocab))]

    st.set_page_config(layout="wide")
    st.title("Visual predictions versus textual labels on Flickr8K")

    col1, _ = st.beta_columns(2)

    col1.markdown("## Options")
    to_group = col1.checkbox("group by image:")
    model_name = col1.selectbox("model: ", MODELS)

    if TO_SHOW_SINGLE_WORD:
        word = col1.selectbox("word: ", words)

    true = load_true(data)
    pred = MODELS[model_name](data)

    def prepare_sample(word_id, sample_id, rank):
        sample = samples[sample_id]
        wav_path = sample["wave"]
        img_path = wav_to_img_path(wav_path)

        key = wav_path_to_key(wav_path)
        transcript = key_to_transcript[key]
        words_sample = [vocab_inv[w] for w in np.where(true[sample_id])[0]]
        description = "\n".join(
            [
                "key:        " + key,
                "rank:       " + str(rank),
                "is-correct: " + str(bool(true[sample_id, word_id])),
                "score:      " + "{:.2f}".format(pred[sample_id, word_id]),
                "labels:     " + ", ".join(words_sample),
                "transcript: " + transcript,
            ]
        )

        return img_path, description

    if to_group:
        # Group groundtruth and predictions corresponding to the same image.
        # Groundtruth is aggergated by the `or` function, while the predictions
        # should all be the same, since they are based only on the input image.
        key_to_key_img = {key: get_key_img(key) for key in keys}
        key_img_to_keys = reverse_non_injective_dict(key_to_key_img)
        keys_img = sorted(key_img_to_keys.keys())

        true, pred = group_true_and_pred(data, true, pred)

        def prepare_sample(word_id, sample_id, rank):
            key_img = keys_img[sample_id]
            img_path = os.path.join(
                config.BASE_DIR,
                "flickr8k-images",
                "Flicker8k_Dataset",
                key_img + ".jpg",
            )

            keys = key_img_to_keys[key_img]
            transcripts = "\n".join("\t" + key_to_transcript[key] for key in keys)
            words_sample = [vocab_inv[w] for w in np.where(true[sample_id])[0]]
            description = "\n".join(
                [
                    "img-key:     " + key_img,
                    "keys:        " + ", ".join(keys),
                    "rank:        " + str(rank),
                    "is-correct:  " + str(bool(true[sample_id, word_id])),
                    "score:       " + "{:.2f}".format(pred[sample_id, word_id]),
                    "labels:      " + ", ".join(words_sample),
                    "transcripts: " + "\n" + transcripts,
                ]
            )

            return img_path, description

    if TO_SHOW_SINGLE_WORD:
        show1(true, pred, vocab, word, prepare_sample)
    else:
        for word in words:
            st.markdown("# " + word)
            show1(true, pred, vocab, word, prepare_sample)
            st.markdown("---")


if __name__ == "__main__":
    main()
