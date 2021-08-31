# TODO
# - [x] evaluate in terms of AUPR (or average precsision) at validation time
# - [x] checkpoint the best model
# - [x] fix random seed
# - [x] run training for `labels-text` and `labels-image-vgg` configurations
# - [x] write script to predict based on a model
# - [x] write script to extract CLIP features
# - [x] (optional) learning rate scheduler
# - [x] train on CLIP features
# - [x] resume from checkpoint
# - [x] AdamW
# - [x] weight decay, but not for biases
# - [x] symmetric loss: match audio to images and images to audio
# - [ ] hyperparameter tunning
# - [ ] log results to wandb
# - [ ] validation in terms of word AUPR
# - [ ] show running average of training error

import os
import pdb
import pickle
import random
import re

from dataclasses import dataclass
from itertools import groupby
from functools import partial

from typing import Any, Dict, List, Literal, Union

import click  # type: ignore

import numpy as np

import torch
import torch.nn

from torch.nn.functional import binary_cross_entropy, cross_entropy, mse_loss
from torch.utils.data import DataLoader, Dataset

from ignite.contrib.metrics import AveragePrecision
from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
    _prepare_batch,
)
from ignite.handlers import (
    ModelCheckpoint,
    create_lr_scheduler_with_warmup,
    CosineAnnealingScheduler,
)
from ignite.metrics import Accuracy, Loss, Metric

import streamlit as st
from matplotlib import pyplot as plt

from toolz import compose, concat, first

from clip.model import Transformer  # type: ignore

import config
from data_gen import spec_augment
from utils import extract_feature, get_soft_tags, get_keywords, get_tran_dict
from scripts.data import get_key_img, load, parse_token


# Types for mypy
Split = Literal["train", "dev", "test"]
TeacherType = Literal["labels-image-vgg", "labels-text", "features-image-clip"]

# Hyper-parameters
HPARAMS: Dict[str, Any] = {
    "batch-size": 64,
    "lr": 4 * 1e-4,
    "num-gradient-updates": 25_000,
    "num-warmup-steps": 1_000,
    "max-len-audio": 2048,
    "seed": 42,
    "weight-decay": 0.01,
}

# Constants
VOCAB_SIZE = 67
EMBED_SIZE = 1000
OUT_DIM = 512
OUTPUT_DIR = "trained_models"

LOG_TRAIN_FREQ = 16
LOG_VALID_FREQ = 256

# Fix random seed
random.seed(HPARAMS["seed"])
np.random.seed(HPARAMS["seed"])
torch.manual_seed(HPARAMS["seed"])


@dataclass
class KeyImage:
    value: str

    def to_key_audio(self, i: int) -> "KeyAudio":
        assert 0 <= i < 5
        return KeyAudio(self.value + "_" + str(i))


@dataclass
class KeyAudio:
    value: str

    def to_key_image(self) -> KeyImage:
        return KeyImage(get_key_img(self.value))


Key = Union[KeyImage, KeyAudio]


class CNNTransformer(torch.nn.Module):
    def __init__(self, embed_dim=512, **kwargs):
        super().__init__()
        # Convolutional module
        width = 128
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(39, 96, 9, 2, 4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(96, 96, 11, 1, 5),
            torch.nn.ReLU(),
            torch.nn.Conv1d(96, 96, 11, 2, 5),
            torch.nn.ReLU(),
            torch.nn.Conv1d(96, 96, 11, 1, 5),
            torch.nn.ReLU(),
            torch.nn.Conv1d(96, 96, 11, 2, 5),
            torch.nn.ReLU(),
            torch.nn.Conv1d(96, 96, 11, 1, 5),
            torch.nn.ReLU(),
            torch.nn.Conv1d(96, width, 11, 2, 5),
        )
        self.transformer = Transformer(width=width, layers=8, heads=4)
        self.ln_final = torch.nn.LayerNorm(width)
        self.proj = torch.nn.Linear(width, embed_dim)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def embed_audio(self, audio):
        x = self.conv(audio)
        x = x.permute(2, 0, 1)  # BDT → TBD
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # TBD → BDT
        x = self.ln_final(x[:, :, 0])
        x = self.proj(x)
        return x

    def forward(self, audio_and_target):
        audio, target = audio_and_target
        input_ = self.embed_audio(audio)
        inp_features = input_ / input_.norm(dim=-1, keepdim=True)
        out_features = target / target.norm(dim=-1, keepdim=True)
        τ = torch.clamp(self.logit_scale.exp(), 0, 100)
        logits = τ * inp_features @ out_features.T
        return logits


AUDIO_MODELS = {
    "cnn-transformer": CNNTransformer,
}


def pad_tensor(vec, pad, dim):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


def pad_collate(batch, dim=0):
    max_len = max(xy[0].shape[dim] for xy in batch)
    xs = torch.stack([pad_tensor(xy[0], max_len, dim) for xy in batch], dim=0)
    ys = torch.stack([xy[1] for xy in batch])
    return xs, ys


class LabelsImageVGGLoader:
    def __init__(self):
        self.soft_tags_dict, self.vocab_soft_all = get_soft_tags(config.soft_tags_fn)
        self.vocab = get_keywords(config.keywords_fn)

    def __call__(self, sample_name):
        return np.array(
            [
                self.soft_tags_dict[sample_name[:-2]][self.vocab_soft_all[word]]
                for word in self.vocab
            ]
        )


class LabelsTextLoader:
    def __init__(self):
        path_tran_dict = os.path.join(config.flickr8k_folder, "tran_dict.pkl")
        with open(path_tran_dict, "rb") as f:
            self.tran_dict = pickle.load(f)
        self.vocab = get_keywords(config.keywords_fn)

    def __call__(self, sample_name: KeyAudio):
        words = self.tran_dict[sample_name.value]
        bow_vector = np.zeros(len(self.vocab)).astype("float32")
        for word in words:
            if word in self.vocab:
                bow_vector[self.vocab[word]] = 1
        return bow_vector


class FeaturesImageCLIPLoader:
    def __init__(self, base_path=""):
        path = os.path.join(base_path, "output/features/clip-ViT-B-16.npz")
        data = np.load(path)
        self.name_to_index = {n: i for i, n in enumerate(data["samples"])}
        self.data = data["data"].astype(np.float32)

    def __call__(self, sample_name: Union[KeyAudio, KeyImage]):
        if isinstance(sample_name, KeyAudio):
            key = sample_name.to_key_image().value
        elif isinstance(sample_name, KeyImage):
            key = sample_name.value
        index = self.name_to_index[key]
        return self.data[index]


class FeaturesAudioCLIPLoader:
    def __init__(self, base_path=""):
        path = os.path.join(
            base_path, "output/cnn-transformer-features-image-clip-flickr8k-test.npz"
        )
        data = np.load(path)
        samples = [s.value for s in Flickr8kDataset.load_samples("test")]
        self.name_to_index = {n: i for i, n in enumerate(samples)}
        self.pred = data["pred"]

    def __call__(self, sample_name: KeyAudio):
        index = self.name_to_index[sample_name.value]
        return self.pred[index]


class FeaturesTextCLIPLoader:
    def __init__(self, base_path=""):
        path = os.path.join(base_path, "output/features/clip-text-ViT-B-16.npz")
        data = np.load(path)
        self.name_to_index = {n: i for i, n in enumerate(data["samples"])}
        self.data = data["data"].astype(np.float32)

    def __call__(self, sample_name: KeyAudio):
        index = self.name_to_index[sample_name.value]
        return self.data[index]


TARGET_LOADERS = {
    "labels-image-vgg": LabelsImageVGGLoader,
    "labels-text": LabelsTextLoader,
    "features-image-clip": FeaturesImageCLIPLoader,
}


class Flickr8kDataset(Dataset):
    def __init__(self, *, split: Split, target_type: TeacherType, is_train: bool):
        super().__init__()
        self.samples = self.load_samples(split)
        self.is_train = is_train
        self.load_target = TARGET_LOADERS[target_type]()

    @staticmethod
    def load_transcripts():
        file_transcript = os.path.join(config.flickr8k_trans_dir, "Flickr8k.token.txt")
        return dict(load(file_transcript, parse_token))

    @staticmethod
    def get_audio_path(sample_name: KeyAudio):
        return os.path.join(
            config.flickr8k_audio_dir, "wavs", sample_name.value + ".wav"
        )

    @staticmethod
    def get_image_path(sample_name: Union[KeyAudio, KeyImage]):
        if isinstance(sample_name, KeyAudio):
            key = sample_name.to_key_image().value
        elif isinstance(sample_name, KeyImage):
            key = sample_name.value
        return os.path.join(
            config.BASE_DIR,
            "flickr8k-images",
            "Flicker8k_Dataset",
            key + ".jpg",
        )

    @staticmethod
    def load_samples_all():
        splits: List[Split] = ["train", "dev", "test"]
        return list(concat(Flickr8kDataset.load_samples(s) for s in splits))

    @staticmethod
    def load_samples(split: Split) -> List[KeyAudio]:
        path_ctm = config.flickr8k_ctm_fn
        path_img = os.path.join(
            config.flickr8k_trans_dir, f"Flickr_8k.{split}Images.txt"
        )

        img_keys = load(path_img, lambda line: line.split(".")[0])
        # Use the CTM to filter the image keys because for some samples we are
        # missing the alignment information.
        keys = list(set(map(first, load(path_ctm, parse_token))))

        img_key_to_keys = {
            k: list(map(KeyAudio, g)) for k, g in groupby(sorted(keys), key=get_key_img)
        }
        return list(concat(img_key_to_keys[img_key] for img_key in img_keys))

    def load_audio_features(self, sample_name):
        feature = extract_feature(
            input_file=self.get_audio_path(sample_name),
            feature="mfcc",
            dim=13,
            cmvn=True,
            delta=True,
            delta_delta=True,
        )
        feature = (feature - feature.mean()) / feature.std()
        if self.is_train:
            feature = spec_augment(feature)
        return feature

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError
        sample_name = self.samples[i]
        audio = torch.tensor(self.load_audio_features(sample_name))
        audio = audio[: HPARAMS["max-len-audio"]]
        audio = audio.t()  # Matches Kayode's model assumption: temporal axis is last
        target = torch.tensor(self.load_target(sample_name))
        return audio, target

    def __len__(self):
        return len(self.samples)


def get_data_loaders(teacher_model_name, batch_size):
    train_loader = DataLoader(
        Flickr8kDataset(split="train", target_type=teacher_model_name, is_train=True),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(pad_collate, dim=1),
    )
    valid_loader = DataLoader(
        Flickr8kDataset(split="dev", target_type=teacher_model_name, is_train=False),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(pad_collate, dim=1),
        drop_last=True,
    )
    return train_loader, valid_loader


def cross_entropy_symmetric(logits, indices):
    loss1 = cross_entropy(logits, indices)
    loss2 = cross_entropy(logits.T, indices)
    return (loss1 + loss2) / 2


def get_mutual_information(loss):
    # From the InfoNCE paper we know that the loss lower-bounds the mutual
    # information, that is I(X; Y) ≥ log(N) - L. This function reports the
    # lower-bound and is useful for comparing models with various batch
    # sizes.
    return np.log(HPARAMS["batch-size"]) - loss


# @click.command()
# @click.option(
#     "-a",
#     "--audio-model",
#     "audio_model_name",
#     required=True,
#     type=click.Choice(AUDIO_MODELS),
# )
# @click.option(
#     "-t",
#     "--teacher",
#     "teacher_model_name",
#     required=True,
#     type=click.Choice(TARGET_LOADERS),
# )
# def main(audio_model_name, teacher_model_name):
#
#     target_type, *_ = teacher_model_name.split("-")
#
#     train_loader, valid_loader = get_data_loaders(
#         teacher_model_name,
#         HPARAMS["batch-size"],
#     )
#
#     if target_type == "features":
#         VOCAB_SIZE = 512
#     else:
#         VOCAB_SIZE = 67
#
#     model = AUDIO_MODELS[audio_model_name](VOCAB_SIZE, EMBED_SIZE)
#     model.to(config.device)
#
#     # if teacher_model_name == "features-image-clip":
#     #     model.τ = torch.nn.Parameter(1.0)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"])
#
#     def batch_cross_entropy(output, target):
#         audio_features = output / output.norm(dim=-1, keepdim=True)
#         image_features = target / target.norm(dim=-1, keepdim=True)
#         logits = audio_features @ image_features.T
#         batch_size, _ = output.shape
#         batch_idxs = torch.arange(batch_size).to(config.device)
#         # print(torch.softmax(logits, dim=0))
#         return cross_entropy(logits, batch_idxs)
#
#     CRITERIA = {
#         "labels": binary_cross_entropy,
#         "features": batch_cross_entropy,
#     }
#
#     def my_criterion(output, target):
#         return CRITERIA[target_type](output[0], target)
#         # return CRITERIA[target_type](output_transform_network(output), target)
#
#     trainer = create_supervised_trainer(
#         model, optimizer, my_criterion, device=config.device
#     )
#
#     valid_metrics: Dict[str, Metric] = {
#         "loss": Loss(my_criterion),
#     }
#
#    def output_transform_network(output):
#        # Assume that the network:
#        # ⅰ. the network outputs extra information, which is discarded;
#        # ⅱ. the predictions correpsond to logits.
#        pred, _ = output
#        return torch.sigmoid(pred)
#
#
#    def output_transform_metric(output_and_true):
#        output, true = output_and_true
#        return output_transform_network(output), true
#
#     if teacher_model_name == "labels-text":
#         valid_metrics["aupr"] = AveragePrecision(
#             output_transform=output_transform_metric
#         )
#
#     evaluator = create_supervised_evaluator(
#         model, metrics=valid_metrics, device=config.device
#     )
#
#     @trainer.on(Events.ITERATION_COMPLETED(every=config.print_freq))
#     def log_training_loss(trainer):
#         print(
#             "train · epoch: {:4d} ◇ loss: {:7.4f}".format(
#                 trainer.state.epoch, trainer.state.output
#             )
#         )
#
#     @trainer.on(Events.EPOCH_COMPLETED)
#     def log_validation_results(trainer):
#         evaluator.run(valid_loader)
#         metrics = evaluator.state.metrics
#         print(
#             "valid · epoch: {:4d} ◇".format(trainer.state.epoch),
#             " ◇ ".join("{:s}: {:7.4f}".format(k, v) for k, v in metrics.items()),
#         )
#
#     # Chekpoint
#     prefix = "{}-{}".format(audio_model_name, teacher_model_name)
#     checkpoint_handler = ModelCheckpoint(
#         OUTPUT_DIR,
#         prefix,
#         n_saved=1,
#         require_empty=False,
#         score_function=lambda engine: -engine.state.metrics["loss"],
#     )
#     evaluator.add_event_handler(
#         event_name=Events.EPOCH_COMPLETED,
#         handler=checkpoint_handler,
#         to_save={"model": model},
#     )
#
#     trainer.run(train_loader, max_epochs=HPARAMS["max-epochs"])


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
@click.option(
    "--checkpoint",
    "checkpoint_path",
    type=click.Path(exists=True),
)
def main(audio_model_name, teacher_model_name, checkpoint_path=None):

    target_type, *_ = teacher_model_name.split("-")

    train_loader, valid_loader = get_data_loaders(
        teacher_model_name,
        HPARAMS["batch-size"],
    )

    model = AUDIO_MODELS[audio_model_name](OUT_DIM)
    model.to(config.device)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])

    named_parameters = list(model.named_parameters())

    def is_gain_or_bias(name):
        subnames = name.split(".")
        bn = re.compile("bn_[0-9]+")
        ln = re.compile("ln_[0-9]+")
        return any(
            bn.match(s) or ln.match(s) or "bias" == s or "logit_scale" == s
            for s in subnames
        )

    params_gain_or_bias = [
        param
        for name, param in named_parameters
        if is_gain_or_bias(name) and param.requires_grad
    ]
    params_other = [
        param
        for name, param in named_parameters
        if not is_gain_or_bias(name) and param.requires_grad
    ]
    group_params = [
        {"params": params_gain_or_bias, "weight_decay": 0.0},
        {"params": params_other, "weight_decay": HPARAMS["weight-decay"]},
    ]
    optimizer = torch.optim.AdamW(group_params, lr=HPARAMS["lr"])

    def prepare_batch(batch, device, non_blocking):
        inp_out = _prepare_batch(batch, device, non_blocking)
        indices = torch.arange(inp_out[0].shape[0]).to(config.device)
        return inp_out, indices

    trainer = create_supervised_trainer(
        model,
        optimizer,
        cross_entropy_symmetric,
        prepare_batch=prepare_batch,
        device=config.device,
    )

    validation_metrics: Dict[str, Metric] = {
        "loss": Loss(cross_entropy_symmetric),
        "accuracy": Accuracy(),
    }

    evaluator = create_supervised_evaluator(
        model,
        metrics=validation_metrics,
        prepare_batch=prepare_batch,
        device=config.device,
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=LOG_TRAIN_FREQ))
    def log_training_loss(trainer):
        print(
            "train · step: {:5d} ◇ loss: {:7.4f} · MI: {:7.4f}".format(
                trainer.state.iteration,
                trainer.state.output,
                get_mutual_information(trainer.state.output),
            )
        )

    @trainer.on(Events.ITERATION_COMPLETED(every=LOG_VALID_FREQ))
    def log_validation_results(trainer):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        print(
            "valid · step: {:5d} ◇ loss: {:7.4f} · MI: {:7.4f} · accuracy: {:.2f}".format(
                trainer.state.iteration,
                metrics["loss"],
                get_mutual_information(metrics["loss"]),
                100 * metrics["accuracy"],
            ),
        )

    # Chekpoint
    prefix = "{}-{}".format(audio_model_name, teacher_model_name)
    checkpoint_handler = ModelCheckpoint(
        OUTPUT_DIR,
        prefix,
        n_saved=5,
        require_empty=False,
        score_function=lambda engine: get_mutual_information(
            engine.state.metrics["loss"]
        ),
    )
    evaluator.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={
            "model": model,
            "optimizer": optimizer,
        },
    )

    cycle_size = HPARAMS["num-gradient-updates"] - HPARAMS["num-warmup-steps"] + 1
    scheduler_cosine = CosineAnnealingScheduler(
        optimizer,
        "lr",
        start_value=HPARAMS["lr"],
        end_value=0,
        cycle_size=cycle_size,
    )
    lr_values = [None] * HPARAMS["num-gradient-updates"]
    scheduler_cosine_warmup = create_lr_scheduler_with_warmup(
        scheduler_cosine,
        warmup_start_value=0.0,
        warmup_end_value=HPARAMS["lr"],
        warmup_duration=HPARAMS["num-warmup-steps"],
        output_simulated_values=lr_values,
    )

    # Plot simulated values
    # lr_values = np.array(lr_values)
    # fig, ax = plt.subplots()
    # ax.plot(lr_values[:, 0], lr_values[:, 1], label="learning rate")
    # st.pyplot(fig)
    # pdb.set_trace()

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_cosine_warmup)

    num_batches = len(train_loader)  # number of gradient updates per epoch
    max_epochs = int(HPARAMS["num-gradient-updates"] / num_batches)
    trainer.run(train_loader, max_epochs=max_epochs)


if __name__ == "__main__":
    main()
