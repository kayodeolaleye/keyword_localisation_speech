# TODO
# - [x] evaluate in terms of AUPR (or average precsision) at validation time
# - [x] checkpoint the best model
# - [x] fix random seed
# - [x] run training for `labels-text` and `labels-image-vgg` configurations
# - [ ] write script to predict based on a model
# - [ ] write script to extract CLIP features
# - [ ] (optional) early stopping
# - [ ] (optional) learning rate scheduler

import os
import pdb
import pickle

from itertools import groupby
from functools import partial

from typing import Any, Dict, List, Literal

import click  # type: ignore

import numpy as np

import torch
import torch.nn

from torch.nn.functional import binary_cross_entropy, mse_loss
from torch.utils.data import DataLoader, Dataset

from ignite.contrib.metrics import AveragePrecision
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, Metric

from toolz import compose, concat, first

import config
from data_gen import spec_augment
from models import MODELS as AUDIO_MODELS
from utils import extract_feature, get_soft_tags, get_keywords, get_tran_dict
from scripts.data import get_key_img, load, parse_token


Split = Literal["train", "dev", "test"]
TeacherType = Literal["labels-image-vgg", "labels-text", "features-image-clip"]


# Hyper-parameters
HPARAMS: Dict[str, Any] = {
    "batch-size": 32,
    "lr": 1e-4,
    "max-epochs": 100,
    "max-len-audio": 2048,
    "seed": 42,
}

VOCAB_SIZE = 67
EMBED_SIZE = 1000
OUTPUT_DIR = "trained_models"


torch.manual_seed(HPARAMS["seed"])


def pad_tensor(vec, pad, dim):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


def pad_collate(batch, dim=0):
    max_len = max(xy[0].shape[dim] for xy in batch)
    xs = torch.stack([pad_tensor(xy[0], max_len, dim) for xy in batch], dim=0)
    ys = torch.stack([xy[1] for xy in batch])
    return xs, ys


def output_transform_network(output):
    # Assume that the network:
    # ⅰ. the network outputs extra information, which is discarded;
    # ⅱ. the predictions correpsond to logits.
    pred, _ = output
    return torch.sigmoid(pred)


def output_transform_metric(output_and_true):
    output, true = output_and_true
    return output_transform_network(output), true


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


class LabelsText:
    def __init__(self):
        path_tran_dict = os.path.join(config.flickr8k_folder, "tran_dict.pkl")
        with open(path_tran_dict, "rb") as f:
            self.tran_dict = pickle.load(f)
        self.vocab = get_keywords(config.keywords_fn)

    def __call__(self, sample_name):
        words = self.tran_dict[sample_name]
        bow_vector = np.zeros(len(self.vocab)).astype("float32")
        for word in words:
            if word in self.vocab:
                bow_vector[self.vocab[word]] = 1
        return bow_vector


TARGET_LOADERS = {
    "labels-image-vgg": LabelsImageVGGLoader,
    "labels-text": LabelsText,
    "features-image-clip": None,
}


class Flickr8kDataset(Dataset):
    def __init__(self, *, split: Split, target_type: TeacherType, is_train: bool):
        super().__init__()
        self.samples = self.load_samples(split)
        self.is_train = is_train
        self.load_target = TARGET_LOADERS[target_type]()

    def load_samples(self, split: Split) -> List[str]:
        path_ctm = config.flickr8k_ctm_fn
        path_img = os.path.join(
            config.flickr8k_trans_dir, f"Flickr_8k.{split}Images.txt"
        )

        img_keys = load(path_img, lambda line: line.split(".")[0])
        # Use the CTM to filter the image keys because for some samples we are
        # missing the alignment information.
        keys = list(set(map(first, load(path_ctm, parse_token))))

        img_key_to_keys = {
            k: list(g) for k, g in groupby(sorted(keys), key=get_key_img)
        }
        return list(concat(img_key_to_keys[img_key] for img_key in img_keys))

    def load_audio_features(self, sample_name):
        audio_path = os.path.join(
            config.flickr8k_audio_dir, "wavs", sample_name + ".wav"
        )
        feature = extract_feature(
            input_file=audio_path,
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
        shuffle=False,
        collate_fn=partial(pad_collate, dim=1),
    )
    return train_loader, valid_loader


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

    target_type, *_ = teacher_model_name.split("-")

    train_loader, valid_loader = get_data_loaders(
        teacher_model_name,
        HPARAMS["batch-size"],
    )

    # TODO use target type information to strip the head
    model = AUDIO_MODELS[audio_model_name](VOCAB_SIZE, EMBED_SIZE)
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"])

    CRITERIA = {
        "labels": binary_cross_entropy,
        "features": mse_loss,
    }

    def my_criterion(output, target):
        return CRITERIA[target_type](output_transform_network(output), target)

    trainer = create_supervised_trainer(
        model, optimizer, my_criterion, device=config.device
    )

    valid_metrics: Dict[str, Metric] = {
        "loss": Loss(my_criterion),
    }

    if teacher_model_name == "labels-text":
        valid_metrics["aupr"] = AveragePrecision(
            output_transform=output_transform_metric
        )

    evaluator = create_supervised_evaluator(
        model, metrics=valid_metrics, device=config.device
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=config.print_freq))
    def log_training_loss(trainer):
        print(
            "train · epoch: {:4d} ◇ loss: {:7.4f}".format(
                trainer.state.epoch, trainer.state.output
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        print(
            "valid · epoch: {:4d} ◇".format(trainer.state.epoch),
            " ◇ ".join("{:s}: {:7.4f}".format(k, v) for k, v in metrics.items()),
        )

    # Chekpoint
    prefix = "{}-{}".format(audio_model_name, teacher_model_name)
    checkpoint_handler = ModelCheckpoint(
        output_dir,
        prefix,
        n_saved=1,
        require_empty=False,
        score_function=lambda engine: -engine.state.metrics["loss"],
    )
    evaluator.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={"model": model},
    )

    trainer.run(train_loader, max_epochs=HPARAMS["max-epochs"])


if __name__ == "__main__":
    main()
