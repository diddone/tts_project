from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 300
    max_seq_len :int = 3000

    encoder_dim:int  = 256
    encoder_n_layer:int  = 4
    encoder_head:int  = 2
    encoder_conv1d_filter_size:int  = 1024

    decoder_dim:int  = 256
    decoder_n_layer:int  = 4
    decoder_head:int  = 2
    decoder_conv1d_filter_size: int = 1024

    fft_conv1d_kernel: tuple = (9, 1)
    fft_conv1d_padding: tuple = (4, 0)

    duration_predictor_filter_size: int = 256
    duration_predictor_kernel_size: int = 3
    dropout: float = 0.1

    PAD: int = 0
    UNK: int = 1
    BOS: int = 2
    EOS: int  = 3

    PAD_WORD: str = '<blank>'
    UNK_WORD: str = '<unk>'
    BOS_WORD: str = '<s>'
    EOS_WORD: str = '</s>'

    # quantization
    n_bins:int = 128
    energy_stats_path: str = 'energy_min_max.npy'
    pitch_stats_path: str = 'pitch_min_max.npy'

