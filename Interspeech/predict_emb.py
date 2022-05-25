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

from sklearn.metrics import average_precision_score

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
    get_dims,
    get_key_img,
    get_mutual_information,
    load_hparams,
    pad_collate,
)


TEACHER_MODEL_NAME = "features-image-clip"
AUDIO_MODEL_NAME = "cnn-transformer"
BATCH_SIZE = 512


def get_score(path):
    _, filename = os.path.split(path)
    filename, _ = os.path.splitext(filename)
    return float(filename.split("_")[-1])


def get_model_path(audio_model_name, teacher_model_name):
    output_dir = os.path.join(OUTPUT_DIR, hparams["name"])
    prefix = "model"
    files = [f for f in os.listdir(output_dir) if f.startswith(prefix)]
    files = sorted(files, key=get_score, reverse=True)
    # assert len(files) == 1

    selected_file = first(files)
    path = os.path.join(OUTPUT_DIR, selected_file)
    return path


# def load_model(audio_model_name, model_path):
#     model = AUDIO_MODELS[audio_model_name](OUT_DIM)
#     model.load_state_dict(torch.load(model_path)["model"])
#     model.to(device)
#     return model


def load_model_hparams(hparams):
    output_dir = os.path.join(OUTPUT_DIR, hparams["name"])
    prefix = "model"

    model = AUDIO_MODELS[hparams["audio-model-name"]](
        hparams["audio-features-size"], **get_dims(hparams)
    )
    files = [f for f in os.listdir(output_dir) if f.startswith(prefix)]
    files = sorted(files, key=get_score, reverse=True)

    model_path = os.path.join(output_dir, files[0])
    model.load_state_dict(torch.load(model_path)["model"])
    model.to(device)
    print("Loaded model from", model_path)

    return model


def predict(hparams: Dict[str, Any]):
    output_path = "output/{}-flickr8k-test.npz".format(hparams["name"])

    if os.path.exists(output_path):
        return

    if hparams["teacher-model-name"] == "features-image-clip":

        def predict1(model, x):
            return model.embed_audio(x)

    elif hparams["teacher-model-name"] in {"labels-image-vgg", "labels-text"}:

        def predict1(model, x):
            return model.forward(x)

    else:
        assert False

    dataset_name = hparams["dataset-name"]
    dataset = DATASETS[dataset_name](
        split="test",
        target_type="dummy",
        audio_features_type=hparams["audio-features-type"],
        to_normalize_audio_features=hparams["audio-features-to-normalize"],
        is_train=False,
    )
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
        indices = torch.arange(len(inp_out[0])).to(device)
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
        device=device,
    )

    pbar = ProgressBar()
    pbar.attach(evaluator)

    evaluator.run(loader)
    metrics = evaluator.state.metrics
    # fmt: off
    print("loss:               {:6.3f} ".format(metrics["loss"]))
    print("mutual information: {:6.3f} ".format(get_mutual_information(metrics["loss"], BATCH_SIZE)))
    print("accuracy:           {:6.3f}%".format(metrics["accuracy"] * 100))
    print()
    # fmt: on


def compute_keyword_spotting_scores_clip(
    config_name: str, vocab, samples: List[Any]
) -> np.ndarray:
    id_to_word = {i: w for w, i in vocab.items()}
    words = [id_to_word[i] for i in range(len(vocab))]

    # encode keywords
    text_template = "a photo of a {word}"
    texts = torch.cat([clip.tokenize(text_template.format(word=w)) for w in words])
    texts = texts.to(device)

    clip_model, preprocess = clip.load("ViT-B/16", device)

    with torch.no_grad():
        text_features = clip_model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.type(torch.float32)
        text_features = text_features.to("cpu")

    # encode audios
    audio_loader = LOADERS["audio-" + config_name]()
    audio_features = torch.vstack([torch.tensor(audio_loader(s)) for s in samples])
    audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

    return audio_features @ text_features.T


def compute_keyword_spotting_scores_labels(
    config_name: str, vocab, samples: List[Any]
) -> np.ndarray:
    loader = LOADERS["audio-" + config_name]()
    predictions = torch.vstack([torch.tensor(loader(s)) for s in samples])
    return predictions


def compute_keyword_spotting_scores(
    config_name: str, vocab, samples: List[Any]
) -> np.ndarray:
    hparams = load_hparams(config_name)
    if hparams["teacher-model-name"] == "features-image-clip":
        f = compute_keyword_spotting_scores_clip
    elif hparams["teacher-model-name"] in {"labels-image-vgg", "labels-text"}:
        f = compute_keyword_spotting_scores_labels
    else:
        assert False
    return f(config_name, vocab, samples)


def eval_keyword_spotting_1(scores, labels, vocab):
    ap = 100 * average_precision_score(labels, scores)
    word_ap = [
        (word, 100 * average_precision_score(labels[:, i], scores[:, i]))
        for word, i in vocab.items()
    ]
    return ap, word_ap


def eval_keyword_spotting(dataset, config_name, vocab_type):
    samples = dataset.samples
    labels_loader = LabelsTextLoader(vocab_type)
    vocab = labels_loader.vocab

    scores = compute_keyword_spotting_scores(config_name, vocab, samples)
    labels = np.vstack([labels_loader(s) for s in samples])

    return eval_keyword_spotting_1(scores, labels, vocab)


LOADERS = {
    "image": FeaturesImageCLIPLoader,
    "text": FeaturesTextCLIPLoader,
}

# for b in [64, 128, 256]:
#     for l in ["", "-lr-2e-4"]:
#         name = f"batch-size-{b}{l}"
#         LOADERS[f"audio-{name}"] = partial(FeaturesAudioCLIPLoader, name=name)

for config in os.listdir("config-files"):
    name, ext = os.path.splitext(config)
    if ext == ".json":
        LOADERS[f"audio-{name}"] = partial(FeaturesAudioCLIPLoader, name=name)


for what in ["logits-cls", "audio-emb"]:
    for λ in "0.01 0.1 10".split():
        name = f"multi-task-lambda-{λ}-{what}"
        LOADERS[f"audio-{name}"] = partial(FeaturesAudioCLIPLoader, name=name)

for what in ["logits-cls", "audio-emb"]:
    name = f"multi-task-{what}"
    LOADERS[f"audio-{name}"] = partial(FeaturesAudioCLIPLoader, name=name)


MODALITIES = ["audio", "image", "text"]


def load_data_retrieval(dataset, src, tgt):

    src_samples = dataset.load_samples("test")
    tgt_samples = dataset.load_samples("test")

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

    src_loader = LOADERS[src]()
    tgt_loader = LOADERS[tgt]()

    src_emb = torch.vstack([torch.tensor(src_loader(s)) for s in src_samples])
    tgt_emb = torch.vstack([torch.tensor(tgt_loader(s)) for s in tgt_samples])

    src_emb = src_emb / src_emb.norm(dim=-1, keepdim=True)
    tgt_emb = tgt_emb / tgt_emb.norm(dim=-1, keepdim=True)

    sim = src_emb @ tgt_emb.T
    sim = sim.numpy()

    return src_samples, tgt_samples, sim, are_same


def evaluate_retrieval(src_samples, tgt_samples, sim, are_same):
    is_correct = np.zeros(sim.shape)
    for i, s in enumerate(tqdm.tqdm(src_samples)):
        top_indices = sim[i].argsort()[::-1]
        is_correct[i] = np.array([are_same(s, tgt_samples[r]) for r in top_indices])
    return {
        "R@{}".format(r): 100 * np.any(is_correct[:, :r], axis=1).mean()
        for r in [1, 5, 10]
    }


VOCAB_CHOICES = "vocab-67-seen vocab-67-unseen vocab-67-unseen-2 vocab-67-unseen-3".split()


@click.command()
@click.option("--config", "config_name")
@click.option("--to-eval-retrieval", "to_eval_retrieval", is_flag=True)
@click.option("--to-eval-keyword-spotting", "to_eval_keyword_spotting", is_flag=True)
@click.option(
    "--vocab",
    "vocab_type",
    type=click.Choice(VOCAB_CHOICES),
    default="vocab-67-seen",
    help="keywords to compute the metrics on",
)
def main(config_name, to_eval_retrieval, to_eval_keyword_spotting, vocab_type):
    # predict and store predictions
    hparams = load_hparams(config_name)
    print(json.dumps(hparams, indent=4))
    output = predict(hparams)

    dataset_name = hparams["dataset-name"]
    dataset = DATASETS[dataset_name](
        split="test",
        target_type="dummy",
        audio_features_type=hparams["audio-features-type"],
        to_normalize_audio_features=hparams["audio-features-to-normalize"],
        is_train=False,
    )

    if to_eval_retrieval:
        audio_modality = f"audio-{config_name}"
        modalities = [audio_modality, "image", "text"]
        results = {
            "`{:5s} → {:5s}`".format(
                src.split("-")[0], tgt.split("-")[0]
            ): evaluate_retrieval(*load_data_retrieval(dataset, src, tgt))
            for src, tgt in product(modalities, modalities)
            if src != tgt and (src == audio_modality or tgt == audio_modality)
        }

        df = pd.DataFrame(results)
        kwargs = {
            "floatfmt": ".1f",
            "tablefmt": "github",
        }
        print(df.transpose().to_markdown(**kwargs))

    if to_eval_keyword_spotting:
        # is there a particular keyword in the audio?
        mean_ap, word_ap = eval_keyword_spotting(dataset, config_name, vocab_type)
        for word, ap in word_ap:
            print("{:10s} {:5.2f}%".format(word, ap))
        print("{:10s} {:5.2f}%".format("mean", mean_ap))
        print()

        # to copy-paste in the google sheets
        s = lambda x: "{:.2f}".format(x)
        ap_all = [mean_ap] + list(map(second, word_ap))
        print(" ".join(map(s, ap_all)))

    if to_eval_retrieval and to_eval_keyword_spotting:
        print(
            "sheets →",
            config_name,
            mean_ap,
            " ".join([str(v) for v in df.unstack().values]),
        )
        print()

    # TODO evaluate mutual information (audio–images)
    # if not model_path:
    #     model_path = get_model_path(AUDIO_MODEL_NAME, TEACHER_MODEL_NAME)
    # model = load_model(AUDIO_MODEL_NAME, model_path)
    # eval_batch(model)
    # eval_keyword_spotting(model, to_store_predictions)


if __name__ == "__main__":
    main()
