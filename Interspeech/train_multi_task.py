import json
import os
import pdb

from typing import Dict
from socket import gethostname

import click
import torch

from ignite.contrib.metrics import AveragePrecision
from ignite.contrib.handlers.wandb_logger import WandBLogger
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
from ignite.metrics.metrics_lambda import MetricsLambda


from train_emb import (
    AUDIO_MODELS,
    HPARAMS,
    OUTPUT_DIR,
    config,
    cross_entropy_symmetric,
    get_data_loaders,
    get_optimizer,
    load_hparams,
    set_random_seed,
)


def train(hparams):
    set_random_seed(hparams["seed"])

    train_loader, valid_loader = get_data_loaders(hparams)

    features_dim = 512
    num_classes = 67

    model = AUDIO_MODELS[hparams["audio-model-name"]](
        hparams["audio-features-size"], embed_dim=features_dim, num_classes=num_classes
    )
    model.to(config.device)

    if hparams.get("checkpoint-path"):
        checkpoint = torch.load(hparams["checkpoint-path"])
        model.load_state_dict(checkpoint["model"])

    optimizer = get_optimizer(
        model, lr=hparams["lr"], weight_decay=hparams["weight-decay"]
    )

    loss_features = cross_entropy_symmetric
    loss_labels = torch.nn.BCEWithLogitsLoss()

    λ = hparams["λ"]

    def loss_fn(pred, true):
        loss1 = loss_features(pred["logits-emb"], true["target-ind"])
        loss2 = loss_labels(pred["logits-clf"], true["target-clf"])
        return loss1 + λ * loss2

    def prepare_batch(batch, device, non_blocking):
        batch = _prepare_batch(batch, device, non_blocking)
        audio, (target_emb, target_clf) = batch
        batch_size = len(audio)
        target_ind = torch.arange(batch_size).to(config.device)
        inp = {
            "audio": audio,
            "target-emb": target_emb,
        }
        out = {
            "target-ind": target_ind,
            "target-clf": target_clf,
        }
        return inp, out

    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss_fn,
        prepare_batch=prepare_batch,
        device=config.device,
    )

    validation_metrics: Dict[str, Metric] = {"loss": Loss(loss_fn)}

    evaluator = create_supervised_evaluator(
        model,
        metrics=validation_metrics,
        prepare_batch=prepare_batch,
        device=config.device,
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=hparams["log-train-freq"]))
    def log_training_loss(trainer):
        print(
            "train · step: {:5d} ◇ loss: {:7.4f}".format(
                trainer.state.iteration,
                trainer.state.output,
            )
        )

    @trainer.on(Events.ITERATION_COMPLETED(every=hparams["log-valid-freq"]))
    def log_validation_results(trainer):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        print(
            "valid · step: {:5d} ◇ loss: {:7.4f}".format(
                trainer.state.iteration,
                metrics["loss"],
            ),
        )

    # Chekpoint
    output_dir = os.path.join(OUTPUT_DIR, hparams["name"])
    prefix = "model"

    checkpoint_handler = ModelCheckpoint(
        output_dir,
        prefix,
        n_saved=5,
        require_empty=False,
        score_function=lambda engine: -engine.state.metrics["loss"],
    )
    evaluator.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={
            "model": model,
            "optimizer": optimizer,
        },
    )

    cycle_size = hparams["num-gradient-steps"] - hparams["num-warmup-steps"] + 1
    scheduler_cosine = CosineAnnealingScheduler(
        optimizer,
        "lr",
        start_value=hparams["lr"],
        end_value=0,
        cycle_size=cycle_size,
    )
    lr_values = [None] * hparams["num-gradient-steps"]
    scheduler_cosine_warmup = create_lr_scheduler_with_warmup(
        scheduler_cosine,
        warmup_start_value=0.0,
        warmup_end_value=hparams["lr"],
        warmup_duration=hparams["num-warmup-steps"],
        output_simulated_values=lr_values,
    )

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_cosine_warmup)

    if hparams["log-wandb"]:
        wandb_logger = WandBLogger(
            project="kayode-train-audio-embedding",
            name=hparams["name"],
            config=hparams,
            tags=[
                "flickr8k",
                hparams["teacher-model-name"],
                "multi-task",
                "machine:" + gethostname(),
            ],
        )

        wandb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED(every=hparams["log-train-freq"]),
            tag="training",
            output_transform=lambda loss: {
                "loss": loss,
            },
        )

        wandb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names=["loss"],
            global_step_transform=lambda *_: trainer.state.iteration,
        )

        wandb_logger.attach_opt_params_handler(
            trainer, event_name=Events.ITERATION_STARTED, optimizer=optimizer
        )

    num_batches = len(train_loader)  # number of gradient updates per epoch
    max_epochs = int(hparams["num-gradient-steps"] / num_batches)
    trainer.run(train_loader, max_epochs=max_epochs)

    if hparams["log-wandb"]:
        wandb_logger.close()


@click.command()
@click.option("--config", "config_name")
def main(config_name=None):
    hparams = load_hparams(config_name)
    print(json.dumps(hparams, indent=4))
    assert hparams["teacher-model-name"] == "features-image-clip+labels-image-vgg"
    assert hparams["audio-model-name"] == "cnn-transformer-multi-task"
    train(hparams)


if __name__ == "__main__":
    main()
