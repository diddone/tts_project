import torch
from torch import nn
from .modules import Transpose
import torch.nn.functional as F
import numpy as np

class Predictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config):
        super(Predictor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = Predictor(model_config)

    @staticmethod
    def create_alignment(base_mat, duration_predictor_output):
        N, L = duration_predictor_output.shape
        for i in range(N):
            count = 0
            for j in range(L):
                for k in range(duration_predictor_output[i][j]):
                    base_mat[i][count+k][j] = 1
                count = count + duration_predictor_output[i][j]
        return base_mat

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = self.create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        log_duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, log_duration_predictor_output
        else:
            duration_predictor_output = (
                (torch.exp(log_duration_predictor_output) - 1)  * alpha + 0.5).clamp(min=0).int()

            output = self.LR(x, duration_predictor_output)

            mel_pos = torch.stack([
                torch.Tensor([i+1 for i in range(output.size(1))])
            ]).long().to('cuda')

            return output, mel_pos


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, model_config):
        super(VarianceAdaptor, self).__init__()


        self.length_regulator = LengthRegulator(model_config)
        self.energy_predictor = Predictor(model_config)
        self.pitch_predictor = Predictor(model_config)

        n_bins = model_config.n_bins
        energy_min, energy_max = np.load(model_config.energy_stats_path)
        pitch_min, pitch_max = np.load(model_config.pitch_stats_path)

        self.energy_bins = nn.Parameter(
            torch.linspace(energy_min, energy_max, n_bins - 1),
            requires_grad=False,
        )
        self.pitch_bins = nn.Parameter(
            torch.linspace(pitch_min, pitch_max, n_bins - 1),
            requires_grad=False,
        )

        self.energy_embedding = nn.Embedding(
            n_bins, model_config.encoder_dim
        )
        self.pitch_embedding = nn.Embedding(
            n_bins, model_config.encoder_dim
        )

    def get_pitch_embedding(self, x, target, control):
        prediction = self.pitch_predictor(x)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, control):
        prediction = self.energy_predictor(x)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        e_target=None,
        p_target=None,
        d_target=None,
        mel_max_length=None,
        e_control=1.0,
        p_control=1.0,
        d_control=1.0,
    ):

        x, log_dur_or_mel_pos = self.length_regulator(x, d_control, d_target, mel_max_length)

        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, e_target, e_control
        )
        x = x + energy_embedding
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, p_target, p_control
        )
        x = x + pitch_embedding

        return (
            x,
            log_dur_or_mel_pos,
            pitch_prediction,
            energy_prediction,
        )

