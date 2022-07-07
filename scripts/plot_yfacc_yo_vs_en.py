import os
import pdb
import pickle
import random
import sys

from matplotlib import pyplot as plt

import numpy as np
import seaborn as sns
import streamlit as st

from adjustText import adjust_text
from toolz import complement, join

from show_results_yfacc import (
    get_word_dict,
    id_to_word_en,
    id_to_word_yo,
    is_localised_word,
    load_data as load_data_yo,
    load_predictions,
    load_samples,
    num_words,
    vocab,
)

from show_results_plos_one import update_text_alignments

sns.set_theme()
θ = 0.5


@st.cache(allow_output_mutation=True)
def load_data_en(model):
    BASE_PATH = "Interspeech"
    sys.path.append(BASE_PATH)
    import config

    def get_key(sample):
        return sample["wave"].split(".")[0].split("/")[-1]
        # key = "_".join(full_key.split("_")[:2])

    with open(os.path.join(BASE_PATH, config.pickle_file), "rb") as f:
        data = pickle.load(f)

    samples = data["test"]
    samples_yo = [s["key"] + "_0" for s in load_samples("test")]
    utt_scores, loc_scores, loc_segments = load_predictions(model)

    samples = update_text_alignments(samples)

    for sample in samples:
        key = get_key(sample)
        sample["key"] = key

        if key not in samples_yo:
            continue

        try:
            sample["utt-score"] = utt_scores[key]
            sample["scores"] = loc_scores[key]
            # print("OK                     ", key)
        except:
            print("WARN missing scores:   ", key)
            continue

        sample["alignment"] = [(lim, word) for lim, _, word in sample["dur"]]

    samples = [sample for sample in samples if "scores" in sample]

    return samples


def is_detected(sample, word_index):
    return sample["utt-score"][word_index] >= θ


def compute_precision_keyword(samples, word_dict):
    word_id = word_dict["id"]

    def get_true_positives():
        return sum(
            is_detected(sample, word_id) and is_localised_word(sample, word_dict)
            for sample in samples
        )

    def get_num_retrieved():
        return sum(is_detected(sample, word_id) for sample in samples)

    pos = get_true_positives()
    num = get_num_retrieved()

    # log
    word = id_to_word_en[word_id]
    print("{:18s} · {:2d} {:2d} ◇ {:5.1f}%".format(word, pos, num, 100 * pos / num))

    return pos / num


def compute_oracle_localisation(samples, word_dict):
    word = word_dict["text"]

    def contains_word(sample):
        return any(w.casefold() == word.casefold() for _, w in sample["alignment"])

    word_id = word_dict["id"]
    samples = [sample for sample in samples if contains_word(sample)]

    def get_num_positives():
        return sum(is_localised_word(sample, word_dict) for sample in samples)

    pos = get_num_positives()
    tot = len(samples)

    if tot == 0:
        score = np.nan
    else:
        score = 100 * pos / tot

    # log
    word = id_to_word_en[word_id]
    print("{:18s} · {:2d} {:2d} ◇ {:5.1f}%".format(word, pos, int(tot), score))

    return {
        "score": score,
        "num-pos": pos,
        "num-tot": tot,
    }


def load_results_yo():
    samples = load_data_yo("yor-5k-init")
    print("yo", len(samples))
    return [
        # compute_precision_keyword(samples, get_word_dict(word_id, "yo"))
        compute_oracle_localisation(samples, get_word_dict(word_id, "yo"))
        for word_id in range(num_words)
    ]


def load_results_en():
    samples = load_data_en("en-5k")
    print("en", len(samples))
    return [
        # compute_precision_keyword(samples, get_word_dict(word_id, "en"))
        compute_oracle_localisation(samples, get_word_dict(word_id, "en"))
        for word_id in range(num_words)
    ]


def plot(word_and_results):
    fig, ax = plt.subplots()

    _, results_en, results_yo = zip(*word_and_results)
    ax.scatter(results_en, results_yo)
    ax.axis("equal")

    ax.set_xlabel("english")
    ax.set_ylabel("yoruba")

    texts = [
        ax.text(x, y, word, ha="center", va="center", size=8)
        for word, x, y in word_and_results
    ]
    adjust_text(
        texts,
        lim=145,
        ha="center",
        arrowprops=dict(arrowstyle="-", color="b", alpha=0.5),
    )
    st.pyplot(fig)

    # plt.tight_layout()
    # plt.savefig("loc-vs-vis.pdf")


def partition(xs, pred):
    return list(filter(pred, xs)), list(filter(complement(pred), xs))


def main():
    # samples_yo = load_data_yo("yor-5k-init")
    # samples_en = load_data_en("en-5k")

    # def contains_word(word, sample):
    #     return any(w.casefold() == word.casefold() for _, w in sample["alignment"])

    # for en, yo in join("key", samples_en, "key", samples_yo):
    #     for w_en, w_yo in vocab:
    #         if contains_word(w_en, en) != contains_word(w_yo, yo):
    #             print(w_en, contains_word(w_en, en))
    #             print(en["alignment"])
    #             print(w_yo, contains_word(w_yo, yo))
    #             print(yo["alignment"])
    #             print()
    #             pdb.set_trace()

    # print(len(set(keys_yo) - set(keys_en)))
    # pdb.set_trace()

    results_yo = load_results_yo()
    results_en = load_results_en()

    word_and_results = [
        ("{}\n{}".format(*words), res_en["score"], res_yo["score"])
        for words, res_en, res_yo in zip(vocab, results_en, results_yo)
        if not np.isnan(res_en["score"])
        and not np.isnan(res_yo["score"])
        and res_en["num-tot"] >= 6
        and res_yo["num-tot"] >= 6
    ]
    # word_and_results_poor, word_and_results_good = partition(
    #     word_and_results, lambda t: t[1] + t[2] <= 30
    # )
    # word_and_results_poor = random.sample(word_and_results_poor, 5)
    # word_and_results = word_and_results_poor + word_and_results_good
    print(len(word_and_results))
    print(word_and_results)
    plot(word_and_results)


if __name__ == "__main__":
    main()
