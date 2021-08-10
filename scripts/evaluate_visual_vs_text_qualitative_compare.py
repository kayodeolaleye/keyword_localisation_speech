import os
import pdb
import pickle

import numpy as np
import streamlit as st

from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_recall_curve

from scripts.evaluate import MODELS, load_true, eval_report, BASE_PATH, config
from scripts.evaluate_visual_vs_text_qualitative import (
    Results,
    ResultsGroupByImage,
    show_top10,
    show_bot5pos,
)


def plot_metrics(axs, true, pred):
    precision, recall, _ = precision_recall_curve(true, pred)
    fpr, tpr, _ = roc_curve(true, pred)

    axs[0].plot(recall, precision)
    axs[0].set_xlabel("recall")
    axs[0].set_ylabel("precision")

    axs[1].plot(fpr, tpr)
    axs[1].set_xlabel("FPR")
    axs[1].set_ylabel("TPR")

    return axs


def show1(word_id, results1, results2):
    true = results1.true

    metrics1 = eval_report(results1.true[:, word_id], results1.pred[:, word_id])
    metrics2 = eval_report(results2.true[:, word_id], results2.pred[:, word_id])

    fig, axs = plt.subplots(ncols=2, figsize=(5.5, 3.0), tight_layout=True)
    axs = plot_metrics(axs, true[:, word_id], results1.pred[:, word_id])
    axs = plot_metrics(axs, true[:, word_id], results2.pred[:, word_id])

    def set_legend(ax, names):
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )

        # Put a legend below current axis
        ax.legend(names, loc="lower center", bbox_to_anchor=(0.5, +1.00))

    def get_names(t):
        return [
            "{} {:.1f}%".format(results1.method_name.split("-")[0], metrics1[t]),
            "{} {:.1f}%".format(results2.method_name.split("-")[0], metrics2[t]),
        ]

    set_legend(axs[0], get_names("aupr"))
    set_legend(axs[1], get_names("auroc"))

    col1, _ = st.beta_columns(2)

    col1.markdown("## Quantitative results")
    col1.pyplot(fig)

    st.markdown("## " + results1.method_name)
    show_top10(word_id, results1)

    st.markdown("## " + results2.method_name)
    show_top10(word_id, results2)


def main(model_name_1="cnn", model_name_2="clip-ViT-B-16-photo-of"):
    with open(os.path.join(BASE_PATH, config.pickle_file), "rb") as f:
        data = pickle.load(f)

    true = load_true(data)
    results1 = ResultsGroupByImage(model_name_1, data, true, MODELS[model_name_1](data))
    results2 = ResultsGroupByImage(model_name_2, data, true, MODELS[model_name_2](data))

    metrics1 = [
        eval_report(results1.true[:, w], results1.pred[:, w])["aupr"]
        for w in range(true.shape[1])
    ]
    metrics2 = [
        eval_report(results2.true[:, w], results2.pred[:, w])["aupr"]
        for w in range(true.shape[1])
    ]

    metrics_diff = np.array(metrics2) - np.array(metrics1)
    ids = metrics_diff.argsort()
    word_ids = np.hstack((ids[:5], ids[-5:]))
    words = sorted(results1.vocab.keys())

    st.set_page_config(layout="wide")
    st.markdown(
        """
    - This page shows qualitative results for two visual methods (`{}` and `{}`) on the test set of the Flickr8K dataset.
    - The predictions of each method are compared to the groundtruth labels extracted from the captions.
    - Since the dataset has the same image captioned multiple times, the predictions and groundtruth are grouped by image id (the groundtruth words are taken as a union across all images, while the predictions should be all the same for the same image).
    - We show results for ten word labels for which the differences (Δ) in AUPR are largest (five for one method, five for the other).
    - For each word label we report AUPR and AUROC, and show the top 10 images ranked by each of the two visual methods.
    """.format(
            model_name_1, model_name_2
        )
    )
    for w in word_ids:
        print(
            "{:10s} {:4.1f} {:4.1f} {:+7.3f}".format(
                words[w], metrics1[w], metrics2[w], metrics_diff[w]
            )
        )
        st.markdown("# Label: {} · Δ(AUPR): {:.2f}%".format(words[w], metrics_diff[w]))
        show1(w, results1, results2)
        st.markdown("---")


if __name__ == "__main__":
    main()
