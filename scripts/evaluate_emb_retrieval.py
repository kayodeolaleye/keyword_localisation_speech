import json
import pdb
import sys

from toolz import identity

import click
import streamlit as st
import numpy as np
import torch
import tqdm

BASE_PATH = "Interspeech"
sys.path.append(BASE_PATH)
from train_emb import (
    FeaturesAudioCLIPLoader,
    FeaturesImageCLIPLoader,
    FeaturesTextCLIPLoader,
    Flickr8kDataset,
    Key,
    KeyAudio,
    KeyImage,
    get_key_img,
)


LOADERS = {
    "audio": FeaturesAudioCLIPLoader,
    "image": FeaturesImageCLIPLoader,
    "text": FeaturesTextCLIPLoader,
}

MODALITIES = LOADERS.keys()


def load_data(src, tgt):

    src_samples = Flickr8kDataset.load_samples("test")
    tgt_samples = Flickr8kDataset.load_samples("test")

    def keys_audio_to_image(keys):
        values = [k.to_key_image().value for k in keys]
        values = sorted(set(values))
        return [KeyImage(value) for value in values]

    if src == "image":
        src_samples = keys_audio_to_image(src_samples)
        tgt_process = get_key_img
    else:
        tgt_process = identity

    if tgt == "image":
        tgt_samples = keys_audio_to_image(tgt_samples)
        src_process = get_key_img
    else:
        src_process = identity

    def are_same(s: Key, t: Key):
        return src_process(s.value) == tgt_process(t.value)

    assert not src == tgt == "image"

    src_loader = LOADERS[src](BASE_PATH)
    tgt_loader = LOADERS[tgt](BASE_PATH)

    src_emb = torch.vstack([torch.tensor(src_loader(s)) for s in src_samples])
    tgt_emb = torch.vstack([torch.tensor(tgt_loader(s)) for s in tgt_samples])

    src_emb = src_emb / src_emb.norm(dim=-1, keepdim=True)
    tgt_emb = tgt_emb / tgt_emb.norm(dim=-1, keepdim=True)

    sim = src_emb @ tgt_emb.T
    sim = sim.numpy()

    return src_samples, tgt_samples, sim, are_same


def evaluate(src_samples, tgt_samples, sim, are_same):
    is_correct = np.zeros(sim.shape)
    for i, s in enumerate(tqdm.tqdm(src_samples)):
        top_indices = sim[i].argsort()[::-1]
        is_correct[i] = np.array([are_same(s, tgt_samples[r]) for r in top_indices])
    return {
        "R@{}".format(r): 100 * np.any(is_correct[:, :r], axis=1).mean()
        for r in [1, 5, 10]
    }


@click.command()
@click.option(
    "-s",
    "--src",
    required=True,
    type=click.Choice(MODALITIES),
    help="source modalitiy",
)
@click.option(
    "-t",
    "--tgt",
    required=True,
    type=click.Choice(MODALITIES),
    help="target modalitiy",
)
def main(src, tgt):
    accuracy = evaluate(*load_data(src, tgt))
    print(json.dumps(accuracy, indent=4))


if __name__ == "__main__":
    main()
