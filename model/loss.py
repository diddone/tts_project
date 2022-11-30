import torch
import torch.nn as nn




import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(
        self, mel,  mel_target,
        log_duration_predicted, duration_predictor_target,
        energy_predicted, energy_predictor_target,
        pitch_predicted, pitch_predictor_target,
        ):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.mse_loss(log_duration_predicted,
                                               (duration_predictor_target.float() + 1.).log())

        energy_predictor_loss = self.mse_loss(energy_predicted, energy_predictor_target)
        pitch_predictor_loss = self.mse_loss(pitch_predicted, pitch_predictor_target)
        return mel_loss, duration_predictor_loss, energy_predictor_loss, pitch_predictor_loss