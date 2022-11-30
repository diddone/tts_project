import torch
from torch.optim.lr_scheduler  import OneCycleLR

from configs import ModelConfig, TrainConfig, MelSpectrogramConfig
from model import FastSpeech, FastSpeechLoss
from utils.wandb_writer import WanDBWriter
from training.train import train
from loaders import get_training_loader
from utils.basic import set_random_seed

def main(train_config: TrainConfig, model_config: ModelConfig, mel_config: MelSpectrogramConfig):

    set_random_seed(train_config.seed)

    training_loader = get_training_loader(train_config)
    model = FastSpeech(model_config, mel_config).to(train_config.device)

    fastspeech_loss = FastSpeechLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9)
    scheduler = OneCycleLR(optimizer, **{
        "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
        "epochs": train_config.epochs,
        "anneal_strategy": "cos",
        "max_lr": train_config.learning_rate,
        "pct_start": 0.1
    })
    logger = WanDBWriter(train_config)


    train(model, optimizer, scheduler, fastspeech_loss, train_config, training_loader, logger)


if __name__ == "__main__":

    mel_config = MelSpectrogramConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()
    main(train_config, model_config, mel_config)

