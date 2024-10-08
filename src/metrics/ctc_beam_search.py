from collections import defaultdict


def expand_and_merge_path(dp, next_token_probs, ind2char, empty_tok):
    new_dp = defaultdict(float)
    for ind, next_token_prob in enumerate(next_token_probs):
        cur_char = ind2char[ind]
        for (prefix, last_char), v in dp.items():
            if cur_char != last_char and cur_char != empty_tok:
                new_dp[(prefix + cur_char, cur_char)] += v * next_token_prob
            else:
                new_dp[(prefix, cur_char)] += v * next_token_prob
    return new_dp


def truncate_paths(dp, beam_size):
    return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])


def ctc_beam_search(log_probs, beam_size, text_encoder):
    dp = {
        ("", text_encoder.EMPTY_TOK): 1.0,
    }
    for prob in log_probs:
        dp = expand_and_merge_path(
            dp, prob, text_encoder.ind2char, text_encoder.EMPTY_TOK
        )
        dp = truncate_paths(dp, beam_size)

    max_value = 0
    max_prefix = ("", "")
    for k, v in dp.items():
        if v > max_value:
            max_value = v
            max_prefix = k
    return max_prefix
