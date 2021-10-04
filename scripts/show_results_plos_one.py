import os
import pdb
import pickle

from collections import Counter
from itertools import groupby

import click
import librosa
import numpy as np
import streamlit as st

from matplotlib import pyplot as plt
from pydub import AudioSegment
from toolz import first, second

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


def load_predictions():
    loc_segments = np.load("Interspeech/output/outputs_204060_ws_untrimmed/full_all_utt_segment_dur.npz", allow_pickle=True)
    loc_scores = np.load("Interspeech/output/outputs_204060_ws_untrimmed/full_all_utt_seg_score.npz", allow_pickle=True)
    utt_scores = np.load("Interspeech/output/outputs_204060_ws_untrimmed/all_full_sigmoid_out.npz", allow_pickle=True)
    # utt_scores = {k: np.max(v, axis=0) for k, v in loc_scores.items()}
    return utt_scores, loc_scores, loc_segments


def plot1(axs, audio, locations, scores, utt_proba, selected_word, text, vocab, rank):
    θ = 0.5

    axs[0].plot(audio)

    loc_best = locations[scores.argmax()]
    # axs[1].axvline(loc_best, linewidth=2, color="green", zorder=0)
    axs[1].bar(locations, scores, width=3)
    axs[1].set_ylim([0, 1])

    axs[0].set_xticks([])
    axs[0].set_yticks([])

    to_sample = lambda v: v / 100 * 16_000

    for (s, e), _, word in text:
        # timelines(0.25, start_end[0], start_end[1], "k", tok)
        axs[0].axvline(to_sample(s), color="gray")
        axs[0].axvline(to_sample(e), color="gray")
        axs[1].axvline(s, color="gray")
        axs[1].axvline(e, color="gray")

    text_locations = [(s + e) / 2 for (s, e), _, _ in text]
    words = [word for _, _, word in text]

    τ = np.argmax(scores)

    def is_in_interval(t):
        return any(s <= t <= e for (s, e), _, w in text if w == selected_word)

    is_detected = any([word == selected_word for word in words])
    is_localised = is_detected and is_in_interval(locations[τ]) and scores[τ] >= θ

    axs[1].set_xticks(text_locations)
    axs[1].set_xticklabels(words, rotation=45)

    # TODO color words based on whether they are keywords or not, for example
    for i, ((start, end), _, _) in enumerate(text):
        if start <= loc_best <= end:
            xticklabel = axs[1].get_xticklabels()[i]
            xticklabel.set_color("#37a9fa")
            xticklabel.set_fontweight("bold")
            axs[1].bar([loc_best], [scores.max()], color="#37a9fa", width=3)

    # axs[1].set_yticks([])
    axs[1].set_ylabel(r"loc. scores α")

    axs[0].set_title(
        "rank: {} · p(w|a) = {:.2f} · is-detected: {} · is-localised: {}".format(
            rank, utt_proba, "✓" if is_detected else "✗", "✓" if is_localised else "✗"
        )
    )


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


@click.command()
@click.option("-p", "--predictions", "predictions_path", required=False)
def main(predictions_path):
    with open(os.path.join(BASE_PATH, config.pickle_file), "rb") as f:
        data = pickle.load(f)

    vocab = data["VOCAB"]
    id_to_word = {i: k for k, i in vocab.items()}

    samples = data["test"]

    if not TO_TRIM:
        samples = update_text_alignments(samples)

    utt_scores, loc_scores, loc_segments = load_predictions()

    word = st.selectbox("keyword", list(vocab.keys()))
    word_id = vocab[word]

    st.markdown("# " + word)
    utt_scores_word = [utt_scores[get_key(sample)][word_id] for sample in samples]
    utt_scores_word = np.array(utt_scores_word)
    top_ids = np.argsort(-utt_scores_word)

    def aggregate_scores(scores, segments):
        # aggregate by midpoint through the maximum operation
        midpoint = lambda seg: (seg[0] + seg[1]) / 2
        agg = lambda group: max(map(second, group))

        locations_scores = sorted(zip(map(midpoint, segments), scores))
        locations_scores_agg = [
            (loc, agg(group)) for loc, group in groupby(locations_scores, first)
        ]
        locations, scores_agg = zip(*locations_scores_agg)

        scores_agg = np.array(scores_agg)
        locations = np.array(locations)
        return scores, locations

    def find_word_at_predicted_location(word_id, sample_id):
        sample = samples[sample_id]
        key = get_key(sample)

        scores = loc_scores[key][:, word_id]
        segments = loc_segments[key]

        scores_agg, locations = aggregate_scores(scores, segments)
        loc_best = locations[np.argmax(scores_agg)]

        text = sample["dur"]

        for i, ((start, end), _, word) in enumerate(text):
            if start <= loc_best <= end:
                return word

    pred_words = [find_word_at_predicted_location(word_id, sample_id) for sample_id in top_ids[:20]]
    counter = Counter(pred_words)
    str_counter = ", ".join("{} ({})".format(w, c) for w, c in counter.most_common(5))

    st.markdown("Top 5 most common words at the predicted location for the top 20 samples:")
    st.code(word.upper() + " → " + str_counter)

    st.markdown("Top 10 samples and their corresponding localisations:")

    for rank, sample_id in enumerate(top_ids[:10], start=1):

        sample = samples[sample_id]
        wav_path = os.path.join(BASE_PATH, sample["wave"])

        key = get_key(sample)

        audio, _ = librosa.load(wav_path, sr=16_000)

        if TO_TRIM:
            audio, _ = librosa.effects.trim(audio, top_db=20)

        scores = loc_scores[key][:, word_id]
        segments = loc_segments[key]

        scores, locations = aggregate_scores(scores, segments)

        durs = sample["dur"]

        fig, axs = plt.subplots(2, 1, figsize=(8, 4))
        plot1(
            axs,
            audio,
            locations,
            scores,
            utt_scores_word[sample_id],
            word.upper(),
            text=durs,
            vocab=vocab,
            rank=rank,
        )

        st.markdown("### key: `{}`".format(key))
        st.pyplot(fig)
        st.audio(wav_path)
        st.markdown("---")


if __name__ == "__main__":
    main()
