import numpy as np
import argparse
from pathlib import Path
from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.utils.io_utils import read_json


def main(dir_path):
    wer = []
    cer = []
    for path in Path(dir_path).iterdir():
        if path.suffix == ".pth":
            text = read_json(path)
            target_text = CTCTextEncoder.normalize_text(text["text"])
            cer.append(calc_cer(target_text, text["pred_text"]))
            wer.append(calc_wer(target_text, text["pred_text"]))
    print("WER:", np.mean(wer))
    print("CER:", np.mean(cer))


if __name__ == "__main__":
    directory = "data/saved/prediction/test"
    main(directory)