import pdb

import numpy as np
import pandas as pd
import streamlit as st

from scripts.show_results_plos_one import load_data, TO_TRIM

samples, vocab = load_data(TO_TRIM)
id_to_word = {i: k for k, i in vocab.items()}
num_words = len(vocab)

def is_localised(sample):
    def is_localised_word(sample, word_id):
        word = id_to_word[word_id]
        scores = sample["scores"][:, word_id]
        locations = sample["locations"]
        text = sample["dur"]
        τ = locations[np.argmax(scores)]
        return any(s <= τ <= e for (s, e), _, w in text if w.lower() == word)
    return np.array([is_localised_word(sample, word_id) for word_id in range(num_words)])

utt_scores = np.vstack([sample["utt-score"] for sample in samples])
is_localised = np.vstack([is_localised(sample) for sample in samples])

st.code("utt_scores =")
utt_scores

st.code("is_localised =")
is_localised

# def compute_precision(word_id, rank=10):
#     idxs = np.argsort(-utt_scores[:, word_id])
#     return 100 * is_localised[idxs, word_id][:rank].sum() / rank

def contains_word(word_id, sample):
    word = id_to_word[word_id]
    return any(word == w.lower() for _, _, w in sample["dur"])

def eval(word_id):
    # sort by detection scores
    idxs = np.argsort(-utt_scores[:, word_id])
    larger_than_thresh = utt_scores[idxs, word_id] >= 0.5
    results = is_localised[idxs, word_id][larger_than_thresh]
    precision = 100 * results.sum() / len(results)
    num_samples_word = sum(1 for sample in samples if contains_word(word_id, sample))
    recall = 100 * results.sum() / num_samples_word
    return 2 * precision * recall / (precision + recall)

p10 = {id_to_word[word_id]: eval(word_id) for word_id in range(num_words)}
p10

np.mean(list(p10.values()))
