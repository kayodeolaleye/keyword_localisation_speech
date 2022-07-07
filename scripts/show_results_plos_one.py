import os
import pdb
import pickle

from collections import Counter
from itertools import groupby

import click
import librosa
import numpy as np
import streamlit as st

# st.set_page_config(layout="wide")

from matplotlib import pyplot as plt
from pydub import AudioSegment
from tabulate import tabulate
from toolz import concat, first, second, partition_all

import matplotlib

# import seaborn as sns

from scripts.data import BASE_PATH, config
from utils_masked import ctm_to_dict, extract_feature, split_frame_length
import config

SEED = 1337
TO_TRIM = False

np.random.seed(SEED)

# matplotlib.rcParams["font.sans-serif"] = "Arial"
# matplotlib.rcParams["font.family"] = "sans-serif"
# sns.set_context("paper")


# def load_predictions(to_trim):
#     suffix = "_untrimmed" if not to_trim else ""
#     base_dir = "Interspeech/output/outputs_204060_ws" + suffix
#     loc_segments = np.load(os.path.join(base_dir, "full_all_utt_segment_dur.npz"))
#     loc_scores = np.load(os.path.join(base_dir, "full_all_utt_seg_score.npz"))
#     utt_scores = np.load(os.path.join(base_dir, "all_full_sigmoid_out.npz"))
#     # utt_scores = {k: np.max(v, axis=0) for k, v in loc_scores.items()}
#     return utt_scores, loc_scores, loc_segments


def load_predictions(to_trim):
    base_dir = "Interspeech/output/outputs_full"
    loc_segments = np.load(os.path.join(base_dir, "full_all_utt_segment_dur_dict.npz"))
    loc_scores = np.load(os.path.join(base_dir, "full_all_utt_seg_score_dict.npz"))
    utt_scores = np.load(os.path.join("Interspeech/output/outputs_204060_ws_untrimmed", "all_full_sigmoid_out.npz"))
    # utt_scores = np.load(os.path.join(base_dir, "full_proposed_max_durations_dict.npz"))
    # utt_scores = {k: np.max(v, axis=0) for k, v in loc_scores.items()}
    return utt_scores, loc_scores, loc_segments


@st.cache(allow_output_mutation=True)
def load_data(to_trim):
    with open(os.path.join(BASE_PATH, config.pickle_file), "rb") as f:
        data = pickle.load(f)

    vocab = data["VOCAB"]
    samples = data["test"]

    utt_scores, loc_scores, loc_segments = load_predictions(TO_TRIM)

    if not to_trim:
        samples = update_text_alignments(samples)

    # augment sample data with classifier's predictions (detection and localisation)
    for sample in samples:
        key = get_key(sample)
        scores, locations = aggregate_scores(loc_scores[key], loc_segments[key])
        sample["key"] = key
        sample["utt-score"] = utt_scores[key]
        sample["locations"] = locations
        sample["scores"] = scores
        sample["text"] = sample["dur"]

    return samples, vocab


def plot_audio(ax, sample, word_selected, query, audio, rank):
    ax.plot(audio)
    ax.set_xlim([0, len(audio)])
    ax.set_xticks([])
    ax.set_yticks([])

    utt_proba = sample["utt-score"][word_selected["id"]]
    locations = sample["locations"]
    scores = sample["scores"][:, word_selected["id"]]
    text = sample["dur"]
    words = [word for _, _, word in text]

    def is_in_interval(t):
        return any(s <= t <= e for (s, e), _, w in text if w == word_selected["text"])

    θ = 0.5
    τ = np.argmax(scores)

    is_detected = any([word == word_selected["text"] for word in words])
    is_localised = is_detected and is_in_interval(locations[τ])

    ax.set_title(
            "query: {} · rank: {} · p(w|a) = {:.2f} · is-detected: {} · is-localised: {}".format(
            query, rank, utt_proba, "✓" if is_detected else "✗", "✓" if is_localised else "✗"
        )
    )

    to_sample = lambda v: v / 100 * 16_000
    from_sample = lambda s: s / 16_000 * 100

    for (s, e), _, word in text:
        ax.axvline(to_sample(s), color="gray")
        ax.axvline(to_sample(e), color="gray")

    s0, s1 = ax.get_xlim()
    xlim = [from_sample(s0), from_sample(s1)]

    return xlim


def plot_predictions(ax, sample, word_selected, vocab, xlim):
    utt_proba = sample["utt-score"][word_selected["id"]]
    locations = sample["locations"]
    scores = sample["scores"][:, word_selected["id"]]
    text = sample["dur"]

    loc_best = locations[scores.argmax()]
    # axs[1].axvline(loc_best, linewidth=2, color="green", zorder=0)
    ax.bar(locations, scores, width=3)
    ax.set_xlim(xlim)
    ax.set_ylim([0, 1])
    ax.set_ylabel(r"loc. scores α")

    for (s, e), _, word in text:
        ax.axvline(s, color="gray")
        ax.axvline(e, color="gray")

    text_locations = [(s + e) / 2 for (s, e), _, _ in text]
    words = [word for _, _, word in text]

    ax.set_xticks(text_locations)
    # ax.set_xticklabels(words, rotation=45)
    ax.set_xticklabels(words)

    # highlight selected word and location
    for i, ((start, end), _, _) in enumerate(text):
        if start <= loc_best <= end:
            xticklabel = ax.get_xticklabels()[i]
            xticklabel.set_color("#37a9fa")
            xticklabel.set_fontweight("bold")
            ax.bar([loc_best], [scores.max()], color="#37a9fa", width=3)


def get_key(sample):
    return sample["wave"].split(".")[0].split("/")[-1]


def update_text_alignments(samples):
    ctm_dict = ctm_to_dict(config.flickr8k_ctm_fn)
    get_key_ctm = lambda key: key[:-2] + ".jpg_#" + key[-1]
    for sample in samples:
        key = get_key_ctm(get_key(sample))
        ctm_entry = ctm_dict[key]
        Δ = 100 * ctm_entry[0][0]
        sample["dur"] = [((s + Δ, e + Δ), d, w) for ((s, e), d, w) in sample["dur"]]
    return samples


def aggregate_scores(scores, segments):
    # aggregate by midpoint through the maximum operation
    midpoint = lambda seg: (seg[0] + seg[1]) / 2
    agg = lambda group: np.max(np.vstack(list(map(second, group))), axis=0)

    locations_scores = sorted(zip(map(midpoint, segments), scores), key=first)
    locations_scores_agg = [
        (loc, agg(group)) for loc, group in groupby(locations_scores, first)
    ]
    locations, scores_agg = zip(*locations_scores_agg)

    scores_agg = np.array(scores_agg)
    locations = np.array(locations)
    return scores_agg, locations


def find_word_at_predicted_location(sample, word_id):
    try:
        loc_best = sample["locations"][np.argmax(sample["scores"][:, word_id])]
    except:
        pdb.set_trace()
    for i, ((start, end), _, word) in enumerate(sample["text"]):
        if start <= loc_best <= end:
            return word


def get_predicted_words_counter(samples, word_id):
    sorted_samples = sorted(samples, key=lambda s: -s["utt-score"][word_id])
    pred_words = [
        find_word_at_predicted_location(sample, word_id)
        for sample in sorted_samples[:20]
    ]
    return Counter(pred_words)


@click.command()
@click.option("-p", "--predictions", "predictions_path", required=False)
def main(predictions_path):
    samples, vocab = load_data(TO_TRIM)
    id_to_word = {i: k for k, i in vocab.items()}

    predicted_words_dict = {
        w: get_predicted_words_counter(samples, vocab[w]) for w in vocab
    }

    def to_str(word):
        pred_words = " · ".join(
            "{:15s} ({:2d})".format(str(w), c)
            for w, c in predicted_words_dict[word].most_common(5)
        )
        return "{:10s} → {}".format(word, pred_words)

    str_predicted_words = "\n".join(to_str(w) for w in vocab)

    # st.code(str_predicted_words)
    def fmt_word(word):
        if word == "<SPOKEN_NOISE>":
            word = "--"
        return word.lower() if word else None
    table_predicted_words = [
        [w] + list(concat([fmt_word(v), c] for v, c in predicted_words_dict[w].most_common(5)))
        for w in vocab
    ]
    st.code(tabulate(table_predicted_words, tablefmt="latex_booktabs"))

    word = st.selectbox("keyword", list(vocab.keys()))
    word_id = vocab[word]
    word_selected = {
        "id": vocab[word],
        "text": word.upper(),
    }

    st.markdown("# " + word)
    counter = predicted_words_dict[word]
    str_counter = ", ".join("{} ({})".format(w, c) for w, c in counter.most_common(5))

    st.markdown(
        "Top 5 most common words at the predicted location for the top 20 samples:"
    )
    st.code(word.upper() + " → " + str_counter)

    st.markdown("Top 12 samples and their corresponding localisations:")

    sorted_samples = sorted(samples, key=lambda s: -s["utt-score"][word_id])
    num_columns = 3

    for ranks_samples in partition_all(num_columns, enumerate(sorted_samples[:12], start=1)):
        cols = st.columns(num_columns)
        for rank, sample in ranks_samples:
            wav_path = os.path.join(BASE_PATH, sample["wave"])
            audio, _ = librosa.load(wav_path, sr=16_000)
            if TO_TRIM:
                audio, _ = librosa.effects.trim(audio, top_db=20)

            width = 8 * len(audio) / 38_000
            fig, axs = plt.subplots(2, 1, figsize=(width, 3.5))
            xlim = plot_audio(axs[0], sample, word_selected, query=word, audio=audio, rank=rank)
            ____ = plot_predictions(axs[1], sample, word_selected, vocab=vocab, xlim=xlim)

            i = int((rank - 1) % num_columns)
            cols[i].markdown("### key: `{}`".format(sample["key"]))
            cols[i].audio(wav_path)
            cols[i].pyplot(fig)

            plt.tight_layout()
            plt.savefig(f"output/plots/plos-one-qualitative-samples/{word}-{rank:02d}.pdf")

        st.markdown("---")


if __name__ == "__main__":
    main()
