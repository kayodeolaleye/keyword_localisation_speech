import pdb

from functools import partial

import numpy as np
import pandas as pd
import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.metrics import average_precision_score

from toolz import second

from adjustText import adjust_text

sns.set_theme(style="white", context="paper", font_scale=1.00)

from scripts.show_results_plos_one import load_data, TO_TRIM


TO_PLOT = [1]


samples, vocab = load_data(TO_TRIM)
id_to_word = {i: k for k, i in vocab.items()}
num_words = len(vocab)


def is_localised(sample):
    def is_localised_word(sample, word_id):
        word = id_to_word[word_id]
        scores = sample["scores"][:, word_id]
        locations = sample["locations"]
        text = sample["dur"]
        τ = locations[np.argmax(scores)]
        return any(s <= τ <= e for (s, e), _, w in text if w.lower() == word)

    return np.array(
        [is_localised_word(sample, word_id) for word_id in range(num_words)]
    )


utt_scores = np.vstack([sample["utt-score"] for sample in samples])
is_localised = np.vstack([is_localised(sample) for sample in samples])

# st.code("utt_scores =")
# utt_scores
#
# st.code("is_localised =")
# is_localised


def contains_word(word_id, sample):
    word = id_to_word[word_id]
    return any(word == w.lower() for _, _, w in sample["dur"])


def evaluate_vis_average_precision(word_id):
    true = has_word[:, word_id]
    pred = vis_scores[:, word_id]
    return 100 * average_precision_score(true, pred)


def evaluate_spotting_precision(word_id, rank=10):
    idxs = np.argsort(-utt_scores[:, word_id])
    return 100 * is_localised[idxs, word_id][:rank].sum() / rank


def evaluate_actual(word_id):
    # sort by detection scores
    idxs = np.argsort(-utt_scores[:, word_id])
    larger_than_thresh = utt_scores[idxs, word_id] >= 0.5
    results = is_localised[idxs, word_id][larger_than_thresh]
    precision = 100 * results.sum() / len(results)
    num_samples_word = sum(1 for sample in samples if contains_word(word_id, sample))
    recall = 100 * results.sum() / num_samples_word
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }


EVAL_METRICS = {
    "actual-f1": lambda w: evaluate_actual(w)["f1"],
    "spotting-p@10": partial(evaluate_spotting_precision, rank=10),
}

METRIC_NAMES = {
    "actual-f1": "F1 score (%)",
    "spotting-p@10": "P@10",
}

metric = "actual-f1"

word_to_result = {
    id_to_word[word_id]: EVAL_METRICS[metric](word_id) for word_id in range(num_words)
}
# results_words
# np.mean(list(p10.values()))

words_results = [(w, r) for w, r in word_to_result.items() if not np.isnan(r)]
words_results = sorted(list(words_results), reverse=True, key=second)
words, results = zip(*words_results)

results = np.array(results)
words = list(words)

print(len(words))


if 1 in TO_PLOT:
    fig, ax = plt.subplots(figsize=(7, 7))
    palette = sns.color_palette("rocket", n_colors=len(words))
    palette.reverse()
    barplot = sns.barplot(x=results, y=words, palette=palette, ax=ax)
    ax.set_xlabel(METRIC_NAMES[metric])

    # add scores
    for i, (w, r) in enumerate(words_results):
        barplot.text(r + 0.2, i, f"{r:.1f}", va="center", fontsize="x-small")

    st.pyplot(fig)

    plt.tight_layout()
    plt.savefig("output/plots/plos-one-keyword-evaluation/f1.pdf")

if 2 in TO_PLOT:

    def repel_labels(ax, x, y, labels, k=0.01):
        G = nx.DiGraph()

        data_nodes = []
        init_pos = {}

        for xi, yi, label in zip(x, y, labels):
            data_str = "data_{0}".format(label)
            G.add_node(data_str)
            G.add_node(label)
            G.add_edge(label, data_str)

            data_nodes.append(data_str)
            init_pos[data_str] = (xi, yi)
            init_pos[label] = (xi, yi)

        pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k)

        # undo spring_layout's rescaling
        pos_after = np.vstack([pos[d] for d in data_nodes])
        pos_before = np.vstack([init_pos[d] for d in data_nodes])

        scale, shift_x = np.polyfit(pos_after[:, 0], pos_before[:, 0], 1)
        scale, shift_y = np.polyfit(pos_after[:, 1], pos_before[:, 1], 1)

        shift = np.array([shift_x, shift_y])

        for key, val in pos.items():
            pos[key] = (val * scale) + shift

        for label, data_str in G.edges():
            ax.annotate(
                label,
                xy=pos[data_str],
                xycoords="data",
                xytext=pos[label],
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-",
                    shrinkA=0,
                    shrinkB=0,
                    connectionstyle="arc3",
                    color="black",
                ),
            )

        # expand limits
        all_pos = np.vstack(pos.values())
        x_span, y_span = np.ptp(all_pos, axis=0)

        mins = np.min(all_pos - x_span * 0.15, 0)
        maxs = np.max(all_pos + y_span * 0.15, 0)

        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[1], maxs[1]])

    # evaluate visual teacher
    vis_scores = np.vstack([sample["soft"] for sample in samples])
    has_word = np.vstack(
        [
            [contains_word(word_id, sample) for word_id in range(num_words)]
            for sample in samples
        ]
    )

    word_to_vis_result = {
        id_to_word[word_id]: evaluate_vis_average_precision(word_id)
        for word_id in range(num_words)
    }
    word_and_results = [
        (word, word_to_result[word], word_to_vis_result[word]) for word in vocab
    ]
    word_and_results = [(w, r, s) for w, r, s in word_and_results if not np.isnan(r)]
    words, results_loc, results_vis = zip(*word_and_results)

    results_loc = np.array(results_loc)
    results_vis = np.array(results_vis)

    # st.code(str(results_loc))
    # st.code(str(results_vis))

    fig, ax = plt.subplots()  # figsize=(5.5, 6))
    ax.scatter(results_loc, results_vis)
    ax.axis("equal")
    ax.set_xlabel("speech network\nactual keyword localisation · F1 score")
    ax.set_ylabel("visual network\nkeyword spotting · average precision")
    texts = [
        ax.text(x, y, word)
        for word, x, y in word_and_results
    ]
    adjust_text(texts, lim=45, arrowprops=dict(arrowstyle="-", color="b", alpha=0.5))
    st.pyplot(fig)

    plt.tight_layout()
    plt.savefig("output/plots/plos-one-keyword-evaluation/loc-vs-vis.pdf")
