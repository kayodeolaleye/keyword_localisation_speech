import click
import os
import pickle
import pdb
import sys

import numpy as np
import pandas as pd

from PIL import Image
from sklearn.metrics import auc, roc_curve, precision_recall_curve

import clip
import torch
import tqdm

BASE_PATH = "Interspeech"
sys.path.append(BASE_PATH)
import config


def load_true(data):
    num_samples = len(data["test"])
    num_vocab = len(data["VOCAB"])

    true = np.zeros((num_samples, num_vocab))
    vocab = {w.casefold(): i for w, i in data["VOCAB"].items()}

    for i, sample in enumerate(data["test"]):
        for word in sample["trn"]:
            try:
                word = word.casefold()
                w = vocab[word]
            except KeyError:
                continue
            true[i, w] = 1

    return true


def load_visual_scores_cnn(data):
    return np.vstack([sample["soft"] for sample in data["test"]])


def load_visual_scores_clip(data):
    path = "output/clip/scores-flickr8-test-keywords57.npy"
    if os.path.exists(path):
        return np.load(path)
    else:
        image_paths = [wav_to_img_path(sample["wave"]) for sample in data["test"]]
        inv_vocab = {i: w for w, i in data["VOCAB"].items()}
        words = [inv_vocab[i] for i in range(len(inv_vocab))]
        scores = compute_visual_scores_clip(image_paths, words)
        os.makedirs("output/clip", exist_ok=True)
        np.save(path, scores)
        return scores


def wav_to_img_path(path):
    _, filename = os.path.split(path)
    filename, _ = os.path.splitext(filename)
    fst, snd, _ = filename.split("_")
    return os.path.join(
        config.BASE_DIR,
        "flickr8k-images",
        "Flicker8k_Dataset",
        fst + "_" + snd + ".jpg",
    )


def compute_visual_scores_clip(image_paths, words, to_visualize=False):
    if to_visualize:
        import streamlit as st

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device)

    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of a {word}") for word in words]
    ).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scores = []
    for image_path in tqdm.tqdm(image_paths):
        image = Image.open(image_path)
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = 100.0 * image_features @ text_features.T

        if to_visualize:
            values, indices = similarity[0].topk(5)
            text = "\n".join(
                f"{words[index]:>16s}: {value.item():.2f}%"
                for value, index in zip(values, indices)
            )
            st.image(image_path)
            st.code(text)
            pdb.set_trace()

        similarity = similarity.cpu().numpy().squeeze(0)
        scores.append(similarity)

    scores = np.vstack(scores)
    return scores


def eval_report(true, pred):
    fpr, tpr, _ = roc_curve(true, pred)
    precision, recall, _ = precision_recall_curve(true, pred)
    return {
        "auroc": 100 * auc(fpr, tpr),
        "aupr": 100 * auc(recall, precision),
    }


VISUAL_MODELS = {
    "cnn": load_visual_scores_cnn,
    "clip": load_visual_scores_clip,
}


@click.command()
@click.option(
    "-vm",
    "--visual-model",
    "visual_model",
    required=True,
    type=click.Choice(list(VISUAL_MODELS.keys())),
)
def main(visual_model):
    with open(os.path.join(BASE_PATH, config.pickle_file), "rb") as f:
        data = pickle.load(f)

    true = load_true(data)
    pred = VISUAL_MODELS[visual_model](data)

    metrics = [eval_report(true[:, w], pred[:, w]) for w in range(true.shape[1])]
    metrics_df = pd.DataFrame(metrics)

    print(metrics_df)
    print(metrics_df.mean(0))


if __name__ == "__main__":
    main()
