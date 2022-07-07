"""Computing the actual localisation precision in Kayode's style."""
import numpy as np

from show_results_yfacc import (
    load_data,
    num_words,
    id_to_word_en,
    is_localised_word,
)


θ = 0.5


def is_detected(sample, word_index):
    return sample["utt-score"][word_index] >= θ


def get_true_positives(word_index):
    return sum(
        is_detected(sample, word_index) and is_localised_word(sample, word_index)
        for sample in samples
    )


def get_num_retrieved(word_index):
    return sum(is_detected(sample, word_index) for sample in samples)


def compute_precision_keyword(word_index):
    pos = get_true_positives(word_index)
    num = get_num_retrieved(word_index)
    word = id_to_word_en[word_index]
    print("{:18s} · {:2d} {:2d} ◇ {:5.1f}%".format(word, pos, num, 100 * pos / num))
    return pos / num


samples = load_data("yor-5k-init")
pos = [get_true_positives(i) for i in range(num_words)]
num = [get_num_retrieved(i) for i in range(num_words)]

print(np.sum(pos))
print(np.sum(num))
print(100 * np.sum(pos) / np.sum(num))
