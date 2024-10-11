import re
from string import ascii_lowercase
import numpy as np

import torch
from pyctcdecode import Alphabet, BeamSearchDecoderCTC
from src.text_encoder.language_model import prepare_language_model
from collections import defaultdict


# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""
    EMPTY_IND = 0

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.decoder = BeamSearchDecoderCTC(Alphabet(self.vocab, False), prepare_language_model())

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char_ind = self.EMPTY_IND
        for ind in inds:
            if last_char_ind == ind:
                continue
            if ind != self.EMPTY_IND:
                decoded.append(self.ind2char[ind])
            last_char_ind = ind

        return "".join(decoded)

    def ctc_beam_search_decode(self, probs) -> str:
        return self.ctc_beam_search(probs)

    def ctc_beam_search_lm_decode(self, probs) -> str:
        return self.decoder.decode(probs, self.beam_size)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    def expand_and_merge_path(self, dp, next_token_probs, ind2char):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_char = ind2char[ind]
            for (prefix, last_char), v in dp.items():
                if cur_char != last_char and cur_char != self.EMPTY_TOK:
                    new_dp[(prefix + cur_char, cur_char)] += v * next_token_prob
                else:
                    new_dp[(prefix, cur_char)] += v * next_token_prob
        return new_dp


    def truncate_paths(self, dp, beam_size):
        return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])


    def ctc_beam_search(self, log_probs):
        dp = {
            ("", self.EMPTY_TOK): 1.0,
        }
        probs = np.exp(log_probs.cpu().detach().numpy())
        for prob in probs:
            dp = self.expand_and_merge_path(dp, prob, self.ind2char)
            dp = self.truncate_paths(dp, self.beam_size)

        max_value = 0
        max_prefix = ("", "")
        for k, v in dp.items():
            if v > max_value:
                max_value = v
                max_prefix = k
        return max_prefix

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size
