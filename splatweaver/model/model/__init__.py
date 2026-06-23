from typing import Optional, Union

from ..encoder import Encoder
from ..encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..encoder.splatweaver import EncoderSplatWeaver, EncoderSplatWeaverCfg
from ..decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from torch import nn
from .splatweaver import SplatWeaver

MODELS = {
    "splatweaver": SplatWeaver,
}

EncoderCfg = Union[EncoderSplatWeaverCfg]
DecoderCfg = DecoderSplattingCUDACfg


# hard code for now
def get_model(encoder_cfg: EncoderCfg, decoder_cfg: DecoderCfg) -> nn.Module:
    model = MODELS['splatweaver'](encoder_cfg, decoder_cfg)
    return model
