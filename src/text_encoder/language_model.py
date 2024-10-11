from pyctcdecode import LanguageModel
import kenlm
import os

def prepare_language_model():
    if os.getcwd().split('/')[1] == 'kaggle':
        model = kenlm.Model("new_dla_asr/language_model/preprocessed_3-gram.pruned.1e-7.arpa")
        with open("new_dla_asr/language_model/librispeech-vocab.txt") as f:
            ngrams = [x.lower() for x in f.read().strip().split("\n")]
    else:
        model = kenlm.Model("language_model/preprocessed_3-gram.pruned.1e-7.arpa")
        with open("language_model/librispeech-vocab.txt") as f:
            ngrams = [x.lower() for x in f.read().strip().split("\n")]
    language_model = LanguageModel(model, unigrams=ngrams)
    return language_model
