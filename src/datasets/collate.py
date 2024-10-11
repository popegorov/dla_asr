import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    audios = []
    specs = []
    texts = []
    texts_enc = []
    audio_paths = []
    spec_lengths = []
    text_enc_lengths = []

    for cur_dict in dataset_items:
        audios.append(torch.transpose(torch.tensor(cur_dict["audio"]), 0, 1))
        spec_lengths.append(cur_dict["spectrogram"].shape[2])
        specs.append(torch.transpose(torch.tensor(cur_dict["spectrogram"]), 0, 2))
        texts.append(cur_dict["text"])  # not to pad
        text_enc_lengths.append(cur_dict["text_encoded"].shape[1])
        texts_enc.append(torch.transpose(torch.tensor(cur_dict["text_encoded"]), 0, 1))
        audio_paths.append(cur_dict["audio_path"])  # not to pad

    result = {}
    result["audio_path"] = audio_paths
    result["text"] = texts
    result["spectrogram_length"] = torch.tensor(spec_lengths)
    result["text_encoded_length"] = torch.tensor(text_enc_lengths)

    result["audio"] = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    result["spectrogram"] = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)
    result["text_encoded"] = torch.nn.utils.rnn.pad_sequence(
        texts_enc, batch_first=True
    )
    result["audio"] = torch.transpose(result["audio"], 1, 2).squeeze(1)
    result["spectrogram"] = torch.transpose(result["spectrogram"], 1, 3).squeeze(1)
    result["text_encoded"] = torch.transpose(result["text_encoded"], 1, 2).squeeze(1)

    return result
