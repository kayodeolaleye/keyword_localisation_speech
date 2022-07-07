import os
import pdb

from toolz import second, partition_all
from matplotlib import pyplot as plt

import librosa
import numpy as np
import streamlit as st
import textgrid

# st.set_page_config(layout="wide")


PATH_DATA = "/home/doneata/data/flickr8k-yoruba/Flickr8k_Yoruba_v6"
PATH_PRED = "Interspeech/output"


def load(path, parser):
    with open(path, "r") as f:
        return list(map(parser, f.readlines()))


def load_vocab():
    path = os.path.join(PATH_DATA, "Flickr8k_text", f"keywords.8_yoruba.txt")

    def parse(line):
        word_en, *words_yo = line.split()
        return word_en, " ".join(words_yo)

    return load(path, parse)


def load_samples(split):
    assert split in "train dev test".split()
    path = os.path.join(
        PATH_DATA, "Flickr8k_text", f"Flickr8k.token.{split}_yoruba.txt"
    )

    def parse(line):
        key, *words = line.split()
        return {
            "key": key.split(".")[0],
            "text": " ".join(words),
        }

    return load(path, parse)


def load_predictions(model):
    loc_scores = np.load(os.path.join(PATH_PRED, model, "all_attention_weight.npz"))
    utt_scores = np.load(os.path.join(PATH_PRED, model, "all_full_sigmoid_out.npz"))
    loc_segments = []  # TODO
    return utt_scores, loc_scores, loc_segments


def load_alignment(key):
    path = os.path.join(PATH_DATA, "Flickr8k_alignment", key + ".TextGrid")
    return [
        ((int(i.minTime * 100), int(i.maxTime * 100)), i.mark.casefold())
        for i in textgrid.TextGrid.fromFile(path)[0]
    ]


@st.cache(allow_output_mutation=True)
def load_data(model):
    samples = load_samples("test")
    utt_scores, loc_scores, loc_segments = load_predictions(model)

    for sample in samples:
        key = sample["key"]
        key = key + "_0"
        sample["key"] = key

        try:
            sample["utt-score"] = utt_scores[key]
            sample["scores"] = loc_scores[key]
            assert loc_scores[key].shape == (67, 800)
        except:
            print("WARN missing scores:   ", key)

        try:
            sample["alignment"] = load_alignment(key)
        except:
            sample["alignment"] = []
            print("WARN missing alignment:", key)

    samples = [sample for sample in samples if "scores" in sample]

    return samples


vocab = load_vocab()
num_words = len(vocab)

id_to_word_en = {i: w for i, (w, _) in enumerate(vocab)}
id_to_word_yo = {i: w for i, (_, w) in enumerate(vocab)}


def get_word_dict(word_id, lang):
    if lang == "en":
        return {
            "id": word_id,
            "lang": lang,
            "text": id_to_word_en[word_id],
        }
    elif lang == "yo":
        return {
            "id": word_id,
            "lang": lang,
            "text": id_to_word_yo[word_id],
        }
    else:
        assert False


def is_localised_word(sample, word_dict):
    word = word_dict["text"].casefold()
    word_id = word_dict["id"]

    scores = sample["scores"][word_id]
    text = sample["alignment"]

    τ = np.argmax(scores)
    is_found = any(s <= τ <= e for (s, e), w in text if w.casefold() == word)

    # print(word, τ, is_found)
    # print(sample["utt-score"][word_id])
    # print(text)
    # print()
    # pdb.set_trace()

    return is_found


def is_localised(sample):
    return np.array(
        [is_localised_word(sample, word_id) for word_id in range(num_words)]
    )


def plot_audio(ax, sample, word_selected, query, audio, rank):
    ax.plot(audio)
    ax.set_xlim([0, len(audio)])
    ax.set_xticks([])
    ax.set_yticks([])

    utt_proba = sample["utt-score"][word_selected["id"]]
    scores = sample["scores"][word_selected["id"]]
    text = sample["alignment"]
    words = [word for _, word in text]

    def is_in_interval(t):
        return any(s <= t <= e for (s, e), w in text if w == word_selected["text"])

    θ = 0.5
    τ = np.argmax(scores)

    is_detected = any([word == word_selected["text"] for word in words])
    is_localised = is_detected and is_in_interval(τ)

    ax.set_title(
        "query: {} · rank: {} · p(w|a) = {:.2f} · is-detected: {} · is-localised: {}".format(
            query,
            rank,
            utt_proba,
            "✓" if is_detected else "✗",
            "✓" if is_localised else "✗",
        )
    )

    to_sample = lambda v: v / 100 * 48_000
    from_sample = lambda s: s / 48_000 * 100

    for (s, e), word in text:
        ax.axvline(to_sample(s), color="gray")
        ax.axvline(to_sample(e), color="gray")

    s0, s1 = ax.get_xlim()
    xlim = [from_sample(s0), from_sample(s1)]

    return xlim


def plot_predictions(ax, sample, word_selected, xlim):
    utt_proba = sample["utt-score"][word_selected["id"]]
    scores = sample["scores"][word_selected["id"]]
    text = sample["alignment"]

    loc_best = scores.argmax()
    # axs[1].axvline(loc_best, linewidth=2, color="green", zorder=0)
    ax.bar(range(len(scores)), scores, width=3)
    ax.set_xlim(xlim)
    ax.set_ylim([0, 1])
    ax.set_ylabel(r"loc. scores α")

    for (s, e), word in text:
        ax.axvline(s, color="gray")
        ax.axvline(e, color="gray")

    text_locations = [(s + e) / 2 for (s, e), _ in text]
    words = [word for _, word in text]

    ax.set_xticks(text_locations)
    # ax.set_xticklabels(words, rotation=45)
    ax.set_xticklabels(words)

    # highlight selected word and location
    for i, ((start, end), _) in enumerate(text):
        if start <= loc_best <= end:
            xticklabel = ax.get_xticklabels()[i]
            xticklabel.set_color("#37a9fa")
            xticklabel.set_fontweight("bold")
            ax.bar([loc_best], [scores.max()], color="#37a9fa", width=3)


def show_samples():
    samples = load_data()
    words_en = [w for w, _ in vocab]

    with st.sidebar:
        word_en = st.selectbox("keyword", words_en)

    index = words_en.index(word_en)
    word_yo = id_to_word_yo[index]

    word_selected = {
        "id": index,
        "text": word_yo,
    }

    st.markdown(f"## Keyword spotting and localisation: `{word_en}` · `{word_yo}`")

    sorted_samples = sorted(samples, key=lambda s: -s["utt-score"][index])
    num_columns = 1
    topk = 10

    for ranks_samples in partition_all(
        num_columns, enumerate(sorted_samples[:10], start=1)
    ):
        cols = st.columns(num_columns)
        for rank, sample in ranks_samples:
            wav_path = os.path.join(
                PATH_DATA,
                "flickr_audio_yoruba_test",
                "S001_" + sample["key"] + "_0" + ".wav",
            )
            audio, _ = librosa.load(wav_path, sr=48_000)

            width = 8 * len(audio) / 180_000
            fig, axs = plt.subplots(2, 1, figsize=(width, 3.5))
            xlim = plot_audio(
                axs[0], sample, word_selected, query=word_yo, audio=audio, rank=rank
            )
            ____ = plot_predictions(axs[1], sample, word_selected, xlim=xlim)

            i = int((rank - 1) % num_columns)
            cols[i].markdown(
                "### rank: {} · key: `{}...`: · score: {:.2f} · is-correct: {}".format(
                    rank,
                    sample["key"][:6],
                    sample["utt-score"][index],
                    "✓" if is_localised_word(sample, index) else "✗",
                )
            )
            cols[i].code(sample["text"])
            cols[i].code(sample["alignment"])
            # cols[i].audio(wav_path)
            cols[i].pyplot(fig)

        st.markdown("---")


def show_evaluate_spotting_localisation():
    samples = load_data()

    utt_scores = np.vstack([sample["utt-score"] for sample in samples])
    is_loc_mat = np.vstack([is_localised(sample) for sample in samples])

    def evaluate_spotting_precision(word_id, rank=10):
        idxs = np.argsort(-utt_scores[:, word_id])
        return 100 * is_loc_mat[idxs, word_id][:rank].sum() / rank

    results = [
        (word_id, evaluate_spotting_precision(word_id)) for word_id in range(num_words)
    ]

    results = [(i, r) for i, r in results if not np.isnan(r)]
    results = sorted(list(results), reverse=True, key=second)

    for i, r in results:
        word_en = id_to_word_en[i]
        word_yo = id_to_word_yo[i]
        print("{:5.1f} ◇ {:15s} · {:20}".format(r, word_en, word_yo))

    print(np.mean([r for _, r in results]))


def main():
    show_samples()
    show_evaluate_spotting_localisation()


if __name__ == "__main__":
    main()
