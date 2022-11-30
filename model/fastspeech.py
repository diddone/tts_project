import torch
from torch import nn
from .encoder_decoder import Encoder, Decoder
from .adaptor import VarianceAdaptor

class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config, mel_config):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, mel_config.num_mels)

        self.model_config = model_config
        self.mel_config = mel_config

    def mask_tensor(self, mel_output, position, mel_max_length):

        def get_mask_from_lengths(lengths, max_len=None):
            if max_len == None:
                max_len = torch.max(lengths).item()

            ids = torch.arange(0, max_len, 1, device=lengths.device)
            mask = (ids < lengths.unsqueeze(1)).bool()

            return mask

        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(
        self, src_seq, src_pos, mel_pos=None,
        e_target=None,
        p_target=None,
        d_target=None,
        mel_max_length=None,
        e_control=1.0,
        p_control=1.0,
        d_control=1.0
        ):

        x, non_pad_mask = self.encoder(src_seq, src_pos)
        if self.training:
            output, log_duration_prediction, energy_prediction, pitch_prediction = self.variance_adaptor(
                x, e_target, p_target, d_target,
                mel_max_length, e_control, p_control, d_control
            )
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)

            output = self.mel_linear(output)
            return output, log_duration_prediction, energy_prediction, pitch_prediction
        else:
            output, mel_pos, pitch_prediction, energy_prediction = self.variance_adaptor(
                x, e_control=e_control, p_control=p_control, d_control=d_control
            )
            output = self.decoder(output, mel_pos)

            output = self.mel_linear(output)
            return output

