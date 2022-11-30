from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import torch
import os

@dataclass
class MelSpectrogramConfig:
    num_mels = 80

@dataclass
class TrainConfig:
    seed: int = 9
    save_dir: str = ''

    checkpoint_path: str = "./model_new"
    logger_path: str = "./logger"
    mel_ground_truth: str = "./mels"
    alignment_path: str = "./alignments"
    energy_path: str = "./energy/"
    pitch_path: str = "./pitch"

    data_path: str = './data/train.txt'

    wandb_project: str = 'test'

    text_cleaners = ['english_cleaners']
    device: str = 'cuda:0'

    batch_size: int = 60
    epochs: int = 200
    n_warm_up_step: int = 4000

    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip_thresh: float = 1.0
    decay_step: tuple = (500000, 1000000, 2000000)

    save_step: int = 3000
    log_step: int = 5
    clear_Time: int = 20

    batch_expand_size:int = 5

    def __post_init__(self):
        if self.seed is None:
            self.seed = np.random.randint(1000000)

        if self.save_dir == '':
            self.save_dir = str(f'/home/dpozdeev/tts_project/save/{datetime.now().strftime(r"%m%d_%H%M%S")}/')

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
