from typing import List
from pyctcdecode import Alphabet, BeamSearchDecoderCTC

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.ctc_beam_search import ctc_beam_search
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class CTCBeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        lengths = log_probs_length.detach().numpy()
        for log_prob, length, target_text in zip(log_probs, lengths, text):
            log_prob_vec = ctc_beam_search(log_prob, self.beam_size, self.text_encoder)[
                0
            ]
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = log_prob_vec[:length]
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)

class NewCTCBeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.decoder = BeamSearchDecoderCTC(Alphabet(text_encoder.vocab, False), None)

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        lengths = log_probs_length.detach().numpy()
        log_probs = log_probs.detach().numpy()
        for log_prob, length, target_text in zip(log_probs, lengths, text):
            pred_text = self.decoder.decode(log_prob[:length])
            target_text = self.text_encoder.normalize_text(target_text)
            wers.append(calc_wer(target_text, pred_text))

        print(sum(wers) / len(wers))
        return sum(wers) / len(wers)
