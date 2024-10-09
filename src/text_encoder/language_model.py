from pyctcdecode import LanguageModel
import kenlm

def prepare_language_model():
    model = kenlm.Model("language_model/lowercase_3-gram.pruned.1e-7.arpa")
    with open("language_model/librispeech-vocab.txt") as f:
        ngrams = [x.lower() for x in f.read().strip().split("\n")]
    language_model = LanguageModel(model, unigrams=ngrams)
    return language_model


