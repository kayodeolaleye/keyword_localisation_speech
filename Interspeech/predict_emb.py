import os
import pdb

from functools import partial
from typing import Dict, Sequence, Tuple

import click
import numpy as np

from toolz import first

import clip

import torch
from torch.utils.data import DataLoader

from ignite.handlers.stores import EpochOutputStore
from ignite.contrib.metrics import AveragePrecision
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events, _prepare_batch, create_supervised_evaluator
from ignite.metrics import Metric

import config

from train_emb import (
    AUDIO_MODELS,
    EMBED_SIZE,
    HPARAMS,
    OUTPUT_DIR,
    OUT_DIM,
    TARGET_LOADERS,
    VOCAB_SIZE,
    Flickr8kDataset,
    # output_transform_metric,
    pad_collate,
)


def load_model(audio_model_name, teacher_model_name):
    prefix = "{}-{}".format(audio_model_name, teacher_model_name)
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(prefix)]
    # assert len(files) == 1

    path = os.path.join(OUTPUT_DIR, first(files))

    model = AUDIO_MODELS[audio_model_name](OUT_DIM)
    model.load_state_dict(torch.load(path)["model"])
    model.to(config.device)

    return model


@click.command()
@click.option(
    "-a",
    "--audio-model",
    "audio_model_name",
    required=True,
    type=click.Choice(AUDIO_MODELS),
)
@click.option(
    "-t",
    "--teacher",
    "teacher_model_name",
    required=True,
    type=click.Choice(TARGET_LOADERS),
)
def main(audio_model_name, teacher_model_name):
    assert teacher_model_name == "features-image-clip" and audio_model_name == "cnn-transformer"

    BATCH_SIZE = 64
    dataset = Flickr8kDataset(split="test", target_type="labels-text", is_train=False)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(pad_collate, dim=1),
    )

    text_template = "a photo of a {word}"
    vocab = dataset.load_target.vocab
    id_to_word = {i: w for w, i in vocab.items()}
    words = [id_to_word[i] for i in range(len(vocab))]

    texts = torch.cat([clip.tokenize(text_template.format(word=w)) for w in words])
    texts = texts.to(config.device)

    clip_model, preprocess = clip.load("ViT-B/16", config.device)

    with torch.no_grad():
        text_features = clip_model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.type(torch.float32)

    def evaluate_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Tuple:
        model.eval()
        with torch.no_grad():
            x, y = _prepare_batch(batch, device=config.device)
            y_pred = model.embed_audio(x)
            return x, y, y_pred

    def output_transform_metric(inp_true_pred):
        _, true, pred = inp_true_pred
        speech_features = pred / pred.norm(dim=-1, keepdim=True)
        similarity = speech_features @ text_features.T
        return similarity, true

    metrics: Dict[str, Metric] = {
        "aupr": AveragePrecision(output_transform_metric),
    }
    model = load_model(audio_model_name, teacher_model_name)

    evaluator = Engine(evaluate_step)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    def output_transform(inp_true_pred):
        _, true, pred = inp_true_pred
        return pred.cpu().numpy(), true.cpu().numpy()

    eos = EpochOutputStore(output_transform)
    eos.attach(evaluator, "output")

    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_results(engine):
        pred = np.vstack([output[0] for output in engine.state.output])
        true = np.vstack([output[1] for output in engine.state.output])
        path = f"output/{audio_model_name}-{teacher_model_name}-flickr8k-test.npy"
        np.savez(path, pred=pred, true=true)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_eval(engine):
        print("aupr: {:.2f}%".format(100 * engine.state.metrics["aupr"]))

    pbar = ProgressBar()
    pbar.attach(evaluator)

    evaluator.run(loader)


if __name__ == "__main__":
    main()
