"""Computing the actual localisation precision in Kayode's style."""
import numpy as np

from show_results_yfacc import (
    get_word_dict,
    load_data as load_data_yo,
    num_words,
    id_to_word_en,
    is_localised_word,
)

from plot_yfacc_yo_vs_en import load_data_en


θ = 0.5
lang = "en"


def is_detected(sample, word_index):
    return sample["utt-score"][word_index] >= θ


def get_true_positives(word_index):
    word_dict = get_word_dict(word_index, lang)
    return sum(
        is_detected(sample, word_index) and is_localised_word(sample, word_dict)
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


if lang == "en":
    samples = load_data_en("en-5k")
elif lang == "yo":
    samples = load_data_yo("yor-5k-init")
else:
    assert False

pos = [get_true_positives(i) for i in range(num_words)]
num = [get_num_retrieved(i) for i in range(num_words)]

print(len(samples))
print(np.sum(pos))
print(np.sum(num))
print(100 * np.sum(pos) / np.sum(num))
