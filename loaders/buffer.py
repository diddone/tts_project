from torch.utils.data import Dataset
from utils.text import process_text, text_to_sequence
import time
from tqdm import tqdm
import os
import numpy as np
import torch


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


def get_data_to_buffer(train_config):
    buffer = list()
    text = process_text(train_config.data_path)

    start = time.perf_counter()
    for i in tqdm(range(len(text))):

        mel_gt_name = os.path.join(
            train_config.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            train_config.alignment_path, str(i)+".npy"))

        energy = torch.load(os.path.join(train_config.energy_path, f'energy_{i}.pt'))
        pitch = torch.load(os.path.join(train_config.pitch_path, f'pitch_{i}.pt'))

        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, train_config.text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        buffer.append({
            "text": character,
            "duration": duration,
            "energy": energy,
            "pitch": pitch,
            "mel_target": mel_gt_target})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer
