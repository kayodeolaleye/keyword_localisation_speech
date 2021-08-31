import os
import pdb

from functools import partial
from typing import Dict, Sequence, Tuple

import click
import numpy as np

from toolz import first

import clip

import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from ignite.handlers.stores import EpochOutputStore
from ignite.contrib.metrics import AveragePrecision
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events, _prepare_batch, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Metric

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
    cross_entropy_symmetric,
    get_mutual_information,
    pad_collate,
)


TEACHER_MODEL_NAME = "features-image-clip"
AUDIO_MODEL_NAME = "cnn-transformer"
BATCH_SIZE = 64


def get_model_path(audio_model_name, teacher_model_name):
    prefix = "{}-{}".format(audio_model_name, teacher_model_name)
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(prefix)]
    # assert len(files) == 1

    selected_file = first(files)
    path = os.path.join(OUTPUT_DIR, selected_file)
    return path


def load_model(audio_model_name, model_path):
    model = AUDIO_MODELS[audio_model_name](OUT_DIM)
    model.load_state_dict(torch.load(model_path)["model"])
    model.to(config.device)
    return model


def eval_batch(model):
    # Perform the same type of evaluation as the one done at train time.
    # Note that this evaluation depends on
    # ⅰ. the batch size and
    # ⅱ. whether the data is shuffled.
    # The larger the batch size the more difficult it is to obtain good scores.
    # If the data is shuffled it easier to obtain good scores.
    dataset = Flickr8kDataset(
        split="test", target_type="features-image-clip", is_train=False
    )

    # On `drop_last` being true.
    # Computing the accuracy needs the same number of classes.
    # Since the number of classes is equal the batch size,
    # we must ensure that all batches have # exactly the same number of samples.
    # To do this, we drop the reminder of samples that do not fit in the whole number of `batch-size`.
    loader = DataLoader(
        dataset,
        batch_size=HPARAMS["batch-size"],
        shuffle=True,
        collate_fn=partial(pad_collate, dim=1),
        drop_last=True,
    )

    def prepare_batch(batch, device, non_blocking):
        inp_out = _prepare_batch(batch, device, non_blocking)
        indices = torch.arange(inp_out[0].shape[0]).to(config.device)
        return inp_out, indices

    def output_transform_acc(output):
        pred, true = output
        return pred.argmax(dim=0), true

    metrics: Dict[str, Metric] = {
        "loss": Loss(cross_entropy_symmetric),
        "accuracy": Accuracy(),
    }

    evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=config.device,
    )

    pbar = ProgressBar()
    pbar.attach(evaluator)

    evaluator.run(loader)
    metrics = evaluator.state.metrics
    print("loss:               {:6.3f} ".format(metrics["loss"]))
    print("mutual information: {:6.3f} ".format(get_mutual_information(metrics["loss"])))
    print("accuracy:           {:6.3f}%".format(metrics["accuracy"] * 100))
    print()


def eval_keyword_spotting(model, to_store_predictions=False):
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

    evaluator = Engine(evaluate_step)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    def output_transform(inp_true_pred):
        _, true, pred = inp_true_pred
        return pred.cpu().numpy(), true.cpu().numpy()

    if to_store_predictions:
        eos = EpochOutputStore(output_transform)
        eos.attach(evaluator, "output")

        @evaluator.on(Events.EPOCH_COMPLETED)
        def save_results(engine):
            pred = np.vstack([output[0] for output in engine.state.output])
            true = np.vstack([output[1] for output in engine.state.output])
            path = f"output/{AUDIO_MODEL_NAME}-{TEACHER_MODEL_NAME}-flickr8k-test.npz"
            np.savez(path, pred=pred, true=true)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_eval(engine):
        print("aupr:      {:.2f}%".format(100 * engine.state.metrics["aupr"]))

    pbar = ProgressBar()
    pbar.attach(evaluator)

    evaluator.run(loader)


@click.command()
@click.option("--model", "model_path", type=click.Path(exists=True))
@click.option("--to-store", "to_store_predictions", is_flag=True)
def main(model_path=None, to_store_predictions=False):
    if not model_path:
        model_path = get_model_path(AUDIO_MODEL_NAME, TEACHER_MODEL_NAME)
    model = load_model(AUDIO_MODEL_NAME, model_path)
    eval_batch(model)
    eval_keyword_spotting(model, to_store_predictions)


if __name__ == "__main__":
    main()
