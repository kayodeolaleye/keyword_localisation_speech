import json
import os
import pdb

from functools import partial
from itertools import product
from typing import Any, Dict, List, Sequence, Tuple

import click
import numpy as np
import pandas as pd
import tqdm

from toolz import first, second, identity

import clip

import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from ignite.handlers.stores import EpochOutputStore
from ignite.contrib.metrics import AveragePrecision
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events, _prepare_batch, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Metric

from config import device

from train_emb import (
    AUDIO_MODELS,
    DATASETS,
    EMBED_SIZE,
    HPARAMS,
    OUTPUT_DIR,
    OUT_DIM,
    TARGET_LOADERS,
    VOCAB_SIZE,
    FeaturesAudioCLIPLoader,
    FeaturesImageCLIPLoader,
    FeaturesTextCLIPLoader,
    Flickr8kDataset,
    Key,
    KeyAudio,
    KeyImage,
    LabelsTextLoader,
    cross_entropy_symmetric,
    get_key_img,
    get_mutual_information,
    load_hparams,
    pad_collate,
)

from predict_emb import (
    VOCAB_CHOICES,
    compute_keyword_spotting_scores_clip,
    compute_keyword_spotting_scores_labels,
    eval_keyword_spotting_1,
    load_model_hparams,
)


BATCH_SIZE = 512


def predict(dataset, hparams: Dict[str, Any], what: str):
    assert hparams["teacher-model-name"] == "features-image-clip+labels-image-vgg"
    assert hparams["audio-model-name"] == "cnn-transformer-multi-task"
    assert what in ("logits-cls", "audio-emb"), "Unknown prediction type"

    output_path = "output/{}-{}-flickr8k-test.npz".format(hparams["name"], what)

    if os.path.exists(output_path):
        return

    if what == "logits-cls":

        def predict1(model, x):
            return model.classify(model.get_emb_shared(x))

    elif what == "audio-emb":

        def predict1(model, x):
            return model.embed_clip(model.get_emb_shared(x))

    samples = [sample.value for sample in dataset.samples]

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(pad_collate, dim=1),
    )

    model = load_model_hparams(hparams)

    def evaluate_step(engine: Engine, batch: Sequence[torch.Tensor]) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            x, _ = _prepare_batch(batch, device=device)
            y_pred = predict1(model, x)
            return y_pred.cpu().numpy()

    evaluator = Engine(evaluate_step)

    pbar = ProgressBar()
    pbar.attach(evaluator)

    eos = EpochOutputStore()
    eos.attach(evaluator, "output")

    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_results(engine):
        pred = np.vstack(engine.state.output)
        np.savez(output_path, pred=pred, samples=samples)

    evaluator.run(loader)


def compute_scores(config_name, vocab, samples, α=0.5):
    scores1 = compute_keyword_spotting_scores_clip(config_name + "-audio-emb", vocab, samples)
    scores2 = compute_keyword_spotting_scores_labels(config_name + "-logits-cls", vocab, samples)
    return α * scores1 + (1 - α) * scores2


@click.command()
@click.option("--config", "config_name")
@click.option(
    "--vocab",
    "vocab_type",
    type=click.Choice(VOCAB_CHOICES),
    default="vocab-67-seen",
    help="keywords to compute the metrics on",
)
def main(config_name, vocab_type):
    # predict and store predictions
    hparams = load_hparams(config_name)
    print(json.dumps(hparams, indent=4))

    dataset_name = hparams["dataset-name"]
    dataset = DATASETS[dataset_name](
        split="test",
        target_type="dummy",
        audio_features_type=hparams["audio-features-type"],
        to_normalize_audio_features=hparams["audio-features-to-normalize"],
        is_train=False,
    )

    predict(dataset, hparams, "logits-cls")
    predict(dataset, hparams, "audio-emb")

    samples = dataset.samples
    labels_loader = LabelsTextLoader(vocab_type)
    vocab = labels_loader.vocab

    if vocab_type.startswith("vocab-67-unseen"):
        α = 1.0

    scores = compute_scores(config_name, vocab, samples, α)
    labels = np.vstack([labels_loader(s) for s in samples])

    mean_ap, word_ap = eval_keyword_spotting_1(scores, labels, vocab)
    for word, ap in word_ap:
        print("{:10s} {:5.2f}%".format(word, ap))
    print("{:10s} {:5.2f}%".format("mean", mean_ap))


if __name__ == "__main__":
    main()
