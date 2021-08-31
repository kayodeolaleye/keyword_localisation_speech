import json
import random
import pdb
import sys

from typing import Union
from toolz import partition

import torch

import numpy as np
import pandas as pd
import streamlit as st

from itertools import product

from scripts.evaluate_emb_retrieval import (
    KeyAudio,
    KeyImage,
    MODALITIES,
    evaluate,
    get_key_img,
    load_data,
)

BASE_PATH = "Interspeech"
sys.path.append(BASE_PATH)
from train_emb import (
    FeaturesImageCLIPLoader,
    FeaturesAudioCLIPLoader,
    Flickr8kDataset,
    get_key_img,
)


def evaluate_all_combinations():
    results = {
        "`{:5s} → {:5s}`".format(src, tgt): evaluate(*load_data(src, tgt))
        for src, tgt in product(MODALITIES, MODALITIES)
        if src != tgt
    }

    df = pd.DataFrame(results)
    kwargs = {
        "floatfmt": ".1f",
        "tablefmt": "github",
    }
    st.code(df.transpose().to_markdown(**kwargs))


def show_retrieval(to_shuffle=True, k=5):
    src = st.selectbox("source modality", MODALITIES, index=0)
    tgt = st.selectbox("target modality", MODALITIES, index=1)

    src_samples, tgt_samples, sim, are_same = load_data(src, tgt)
    transcripts = Flickr8kDataset.load_transcripts()

    def get_text(sample: Union[KeyAudio, KeyImage]):
        if isinstance(sample, KeyAudio):
            return transcripts[sample.value]
        elif isinstance(sample, KeyImage):
            return "\n" + "\n".join(
                "\t" + transcripts[sample.to_key_audio(i).value] for i in range(5)
            )

    st.markdown(f"## retrieval · {src} → {tgt}")
    st.markdown(
        f"""
        - for each {src} sample in the test set, we retrieve the top {k} most similar {tgt} samples
        - use audio model trained in the embedding space with images represented by CLIP features
        - below we show 16 random samples
    """
    )

    indices = list(range(len(src_samples)))
    if to_shuffle:
        random.shuffle(indices)
    indices = indices[:16]

    for i in indices:
        s = src_samples[i]
        top_indices = sim[i].argsort()[::-1][:k]
        is_correct = any(are_same(s, tgt_samples[r]) for r in top_indices)

        desc = [
            f"in-top-{k}: " + str(is_correct),
            f"text:       " + get_text(s),
        ]

        col, *_ = st.beta_columns([0.4, 0.6])
        col.markdown("## " + s.value)
        col.image(Flickr8kDataset.get_image_path(s))
        if src != "image":
            col.audio(Flickr8kDataset.get_audio_path(s))
        col.code("\n".join(desc))

        st.markdown(f"top {k} {tgt} samples by {src} query")
        for group in partition(5, top_indices):
            cols = st.beta_columns(5)
            for r, col in zip(group, cols):
                t = tgt_samples[r]
                is_correct = are_same(s, t)
                desc = [
                    "sample:     " + t.value,
                    "score:      {:.3f}".format(sim[i, r]),
                    "is-correct: " + str(is_correct),
                    "transcripts:" + get_text(t),
                ]
                col.image(Flickr8kDataset.get_image_path(t))
                col.code("\n".join(desc))

        st.markdown("---")


def main():
    st.set_page_config(layout="wide")
    # evaluate_all_combinations()
    show_retrieval()


if __name__ == "__main__":
    main()
