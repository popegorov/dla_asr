from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer


# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class CTCBeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        lengths = log_probs_length.cpu().detach().numpy()
        for log_prob, length, target_text in zip(log_probs, lengths, text):
            pred_text = self.text_encoder.ctc_beam_search_decode(log_prob[:length, :])[0]
            target_text = self.text_encoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)

class CTCBeamSearchLMCERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        lengths = log_probs_length.cpu().detach().numpy()
        log_probs = log_probs.cpu().detach().numpy()
        for log_prob, length, target_text in zip(log_probs, lengths, text):
            pred_text = self.text_encoder.ctc_beam_search_lm_decode(log_prob[:length, :])
            target_text = self.text_encoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)

