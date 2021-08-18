import os
import pdb

from functools import partial
from typing import Dict

import click
import numpy as np

from toolz import first

import torch
from torch.utils.data import DataLoader

from ignite.handlers.stores import EpochOutputStore
from ignite.contrib.metrics import AveragePrecision
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Metric

import config

from train_emb import (
    AUDIO_MODELS,
    EMBED_SIZE,
    HPARAMS,
    OUTPUT_DIR,
    TARGET_LOADERS,
    VOCAB_SIZE,
    Flickr8kDataset,
    output_transform_metric,
    pad_collate,
)


def load_model(audio_model_name, teacher_model_name):
    prefix = "{}-{}".format(audio_model_name, teacher_model_name)
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(prefix)]
    assert len(files) == 1

    path = os.path.join(OUTPUT_DIR, first(files))

    model = AUDIO_MODELS[audio_model_name](VOCAB_SIZE, EMBED_SIZE)
    model.load_state_dict(torch.load(path))
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
    BATCH_SIZE = 256
    loader = DataLoader(
        Flickr8kDataset(split="test", target_type="labels-text", is_train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(pad_collate, dim=1),
    )

    metrics: Dict[str, Metric] = {
        "aupr": AveragePrecision(output_transform=output_transform_metric),
    }
    model = load_model(audio_model_name, teacher_model_name)
    evaluator = create_supervised_evaluator(model, metrics, device=config.device)

    def output_transform(output_and_true):
        output, true = output_and_true
        pred, _ = output
        return true, pred

    eos = EpochOutputStore(output_transform=output_transform)
    eos.attach(evaluator, "output")

    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_results(engine):
        pred = np.vstack([output[0].cpu() for output in engine.state.output])
        true = np.vstack([output[1].cpu() for output in engine.state.output])
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
