"""Analysis on the YFACC results provided by Kayode."""
import os
import pdb
import pickle
import random
import sys

from matplotlib import pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

import pandas
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
    """Precision of actual keyword localisation."""
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
    word_id = word_dict["id"]

    samples = [sample for sample in samples if contains_word(sample, word)]

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


def compute_detection_f1(samples, word_dict):
    word = word_dict["text"]
    word_id = word_dict["id"]

    true = [contains_word(sample, word) for sample in samples]
    pred = [is_detected(sample, word_id) for sample in samples]

    score = 100 * average_precision_score(true, pred)

    # log
    word = id_to_word_en[word_id]
    print("{:18s} ◇ {:5.1f}%".format(word, score))

    return {
        "score": score,
        "num-true": sum(true),
        "num-pred": sum(pred),
    }


def load_results_yo():
    samples = load_data_yo("yor-5k-init")
    print("yo", len(samples))
    return [
        # compute_precision_keyword(samples, get_word_dict(word_id, "yo"))
        # compute_oracle_localisation(samples, get_word_dict(word_id, "yo"))
        compute_detection_f1(samples, get_word_dict(word_id, "yo"))
        for word_id in range(num_words)
    ]


def load_results_en():
    samples = load_data_en("en-5k")
    print("en", len(samples))
    return [
        # compute_precision_keyword(samples, get_word_dict(word_id, "en"))
        # compute_oracle_localisation(samples, get_word_dict(word_id, "en"))
        compute_detection_f1(samples, get_word_dict(word_id, "en"))
        for word_id in range(num_words)
    ]


def plot(word_and_results):
    # remove nan's
    word_and_results = [
        (w, e, y) for w, e, y in word_and_results if not np.isnan(e) and not np.isnan(y)
    ]
    _, results_en, results_yo = zip(*word_and_results)

    fig, ax = plt.subplots()
    ax.plot([0, 100], [0, 100], linestyle="--", color="gray", alpha=0.5)
    ax.scatter(results_en, results_yo)
    ax.axis("equal")

    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])

    ax.set_xlabel("english")
    ax.set_ylabel("yoruba")

    texts = [
        ax.text(x, y, word, ha="center", va="center", size=8)
        for word, x, y in word_and_results
    ]
    adjust_text(
        texts,
        lim=45,
        ha="center",
        arrowprops=dict(arrowstyle="-", color="b", alpha=0.5),
    )
    st.pyplot(fig)

    # plt.tight_layout()
    # plt.savefig("loc-vs-vis.pdf")


def contains_word(sample, word):
    return any(w.casefold() == word.casefold() for _, w in sample["alignment"])


def partition(xs, pred):
    return list(filter(pred, xs)), list(filter(complement(pred), xs))


def show_cross_lingual_differences():
    samples_yo = load_data_yo("yor-5k-init")
    samples_en = load_data_en("en-5k")

    def to_text(a):
        return " ".join(w.casefold() for _, w in a if w)

    for en, yo in join("key", samples_en, "key", samples_yo):
        for w_en, w_yo in vocab:
            if w_en != "football":
                continue
            # if contains_word(en, w_en) != contains_word(yo, w_yo):
            if contains_word(en, w_en) or contains_word(yo, w_yo):
                # print(w_en, contains_word(en, w_en))
                print(en["key"])
                print(to_text(en["alignment"]))
                # print(w_yo, contains_word(yo, w_yo))
                print(to_text(yo["alignment"]))
                print()
                # pdb.set_trace()

    # print(len(set(keys_yo) - set(keys_en)))
    # pdb.set_trace()


def show_cross_lingual_visual_teacher():
    description = """
    ## Cross-lingual performance of the visual teacher

    Compute the performance of the visual teacher on the English and Yoruba datasets.
    The images are the same in both cases (so the visual teacher's predictions are the same),
    but the captions are different.
    Even if they are translations of each there are differences.
    Report mean average precision.

    - football / soccer use the same Yoruba keyword, but visually distinct
    - beach / ocean use also th same Yoruba keyword, but we expect to be correlated visually
    - words like mountain, bike are more frequent in the Yoruba (maybe because bicycle, top) are mapped to them so the base performance is higher.

    """
    st.markdown(description)

    samples_yo = load_data_yo("yor-5k-init")
    samples_en = load_data_en("en-5k")

    words_en, words_yo = zip(*vocab)

    def to_binary(sample, words):
        return [int(contains_word(sample, word)) for word in words]

    def prepare1(en, yo):
        pred_vis = en["soft"]
        true_en = to_binary(en, words_en)
        true_yo = to_binary(yo, words_yo)
        return pred_vis, true_en, true_yo

    data = [prepare1(en, yo) for en, yo in join("key", samples_en, "key", samples_yo)]
    pred_vis, true_en, true_yo = zip(*data)

    pred_vis = np.vstack(pred_vis)
    true_en = np.vstack(true_en)
    true_yo = np.vstack(true_yo)

    res_en = [
        100 * average_precision_score(true_en[:, w], pred_vis[:, w])
        for w in range(num_words)
    ]
    res_yo = [
        100 * average_precision_score(true_yo[:, w], pred_vis[:, w])
        for w in range(num_words)
    ]

    num_en = [sum(true_en[:, w]) for w in range(num_words)]
    num_yo = [sum(true_yo[:, w]) for w in range(num_words)]
    texts = ["{}\n{} · {}".format(w, e, y) for w, e, y in zip(words_en, num_en, num_yo)]
    # texts = words_en

    word_and_results = zip(texts, res_en, res_yo)
    plot(word_and_results)


def show_cross_lingual_keyword_correlation():
    def cohen_kappa_score_max(y1, y2):
        confusion = confusion_matrix(y1, y2)
        n_classes = confusion.shape[0]
        assert n_classes == 2

        p = confusion / confusion.sum()
        p_row = p.sum(axis=0)
        p_col = p.sum(axis=1)

        # p_max = p[0, 0] + p[1, 1]
        p_max = np.sum(np.minimum(p_row, p_col))
        p_exp = np.sum(p_row * p_col)
        return (p_max - p_exp) / (1 - p_exp)

    def cohen_kappa_score_within(y1, y2):
        return cohen_kappa_score(y1, y2) / cohen_kappa_score_max(y1, y2)

    def argsort_sim_mat(sm):
        idx = [np.argmax(np.sum(sm, axis=1))]  # a
        for i in range(1, len(sm)):
            sm_i = sm[idx[-1]].copy()
            sm_i[idx] = -1
            idx.append(np.argmax(sm_i))  # b
        return np.array(idx)

    def get_sim_mat2(sm):
        n = sm.shape[0]
        return 1 / (np.linalg.norm(sm[:, np.newaxis] - sm[np.newaxis], axis=-1) + 1 / n)

    description = """
    ## Cross-lingual keyword correlation

    For each pair of English–Yoruba keyword pair, show the correlation across the test samples.
    Report the normalize Kappa score.

    """
    st.markdown(description)

    samples_yo = load_data_yo("yor-5k-init")
    samples_en = load_data_en("en-5k")

    words_en, words_yo = zip(*vocab)
    # words_yo = list(set(words_yo))  # unique words

    # Remove nans
    # idxs_nan = [20, 21, 44, 49, 54]
    # idxs = set(range(num_words)) - set(idxs_nan)
    # idxs = random.sample(idxs, 20)

    selected_words = [
        "beach",
        "ocean",
        "sand",
        "small",
        "little",
        "stick",
        "tree",
        "white",
        "red",
        "tennis",
        "ball",
        "stands",
        "sits",
        "football",
        "soccer",
        "shirt",
        "wearing",
        # "water",
        "boy",
        "mountain",
        "top",
    ]
    idxs = [words_en.index(w) for w in selected_words]

    words_en = [w for i, w in enumerate(words_en) if i in idxs]
    words_yo = [w for i, w in enumerate(words_yo) if i in idxs]

    def to_binary(sample, words):
        return [int(contains_word(sample, word)) for word in words]

    def prepare1(en, yo):
        true_en = to_binary(en, words_en)
        true_yo = to_binary(yo, words_yo)
        return true_en, true_yo

    data = [prepare1(en, yo) for en, yo in join("key", samples_en, "key", samples_yo)]

    true_en, true_yo = zip(*data)
    true_en = np.vstack(true_en)
    true_yo = np.vstack(true_yo)

    sm = np.vstack(
        [
            [
                cohen_kappa_score_within(true_en[:, i], true_yo[:, j])
                for i, en in enumerate(words_en)
            ]
            for j, yo in enumerate(words_yo)
        ]
    )

    # scores = sm.sum(0) + sm.sum(1)
    # scores = true_en.sum(0) + true_yo.sum(0)
    # print(list(zip(scores, words_en)))
    # idxs = sorted(np.argsort(scores)[-20:])
    # sm = sm[idxs, :][:, idxs]
    # words_en_1 = [words_en[i] for i in idxs]
    # words_yo_1 = [words_yo[i] for i in idxs]

    idxs = argsort_sim_mat(get_sim_mat2(sm))
    sm = sm[idxs, :][:, idxs]

    words_en_1 = [words_en[i] for i in idxs]
    words_yo_1 = [words_yo[i] for i in idxs]

    df = pandas.DataFrame(sm, index=words_yo_1, columns=words_en_1)
    df = df[
        ~df.index.duplicated(keep="first")
    ]  # some words in Yoruba appears multiple times
    df = df.stack().reset_index(name="κ score")
    df = df.rename(columns={"level_0": "yo", "level_1": "en"})
    df

    # df = pandas.DataFrame(
    #     [
    #         {
    #             "en": en,
    #             "yo": yo,
    #             "κ score": cohen_kappa_score_within(true_en[:, i], true_yo[:, j]),
    #         }
    #         for i, en in enumerate(words_en)
    #         for j, yo in enumerate(words_yo)
    #         if sum(true_yo[:, j]) > 0
    #     ]
    # )

    # df = df.pivot(index="en", columns="yo", values="κ score")
    # df

    with sns.axes_style(style="whitegrid"):
        g = sns.relplot(
            data=df,
            x="yo",
            y="en",
            hue="κ score",
            size="κ score",
            palette="vlag",
            edgecolor=".7",
            hue_norm=(-1, 1),
            height=4,
            sizes=(5, 100),
            size_norm=(-0.1, 1.0),
        )

        # Tweak the figure to finalize
        g.set(aspect="equal")
        g.despine(left=True, bottom=True)
        g.ax.margins(0.04)

        for label in g.ax.get_xticklabels():
            label.set_rotation(90)

        for artist in g.legend.legendHandles:
            artist.set_edgecolor(".7")

        st.pyplot(g)

        # plt.tight_layout()
        g.savefig("output/plots/yfacc/cross-lingual-keyword-correlation.pdf")


def show_yo_vs_en_keyword_performance():
    results_yo = load_results_yo()
    results_en = load_results_en()

    word_and_results = [
        ("{}\n{}".format(*words), res_en["score"], res_yo["score"])
        for words, res_en, res_yo in zip(vocab, results_en, results_yo)
        if not np.isnan(res_en["score"])
        and not np.isnan(res_yo["score"])
        and res_en["num-true"] >= 6
        and res_yo["num-true"] >= 6
    ]
    word_and_results_poor, word_and_results_good = partition(
        word_and_results, lambda t: t[1] + t[2] <= 30
    )
    # word_and_results_poor = random.sample(word_and_results_poor, len(word_and_results_poor) // 2)
    word_and_results = word_and_results_poor + word_and_results_good
    print(len(word_and_results))
    print(word_and_results)
    plot(word_and_results)


def main():
    show_cross_lingual_differences()
    show_cross_lingual_keyword_correlation()
    # show_cross_lingual_visual_teacher()


if __name__ == "__main__":
    main()
