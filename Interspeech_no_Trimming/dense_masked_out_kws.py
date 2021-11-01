from librosa.core.audio import get_duration
import numpy as np
import torch
import os
from tqdm import tqdm
import argparse
from utils import ensure_folder, extract_feature, get_gt_token_duration, get_logger, get_localisation_metric_count, eval_localisation_accuracy, eval_localisation_prf, split_frame_length, plot_stuff
from os import path
from config import pickle_file, device, trained_model_dir
import pickle
from utils import parse_args, get_token_dur_dict
import sklearn.metrics as metrics
from scipy.interpolate import interp1d
from scipy.optimize import brentq

def parse_args():
    parser = argparse.ArgumentParser(description='dense keyword spotting and localisation in speech')
    parser.add_argument('--target_type', type=str, help='provide the type of target to use for supervision')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument("--analyze", help="print an analysis of the evaluation output for each utterance", action="store_true")
    args = parser.parse_args()
    return args

def pad(feature):
    max_input_len = 800
    input_length = feature.shape[0]
    input_dim = feature.shape[1]
    padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
    length = min(input_length, max_input_len)
    padded_input[:length, :] = feature[:length, :]

    padded_input = np.transpose(padded_input, (1, 0))

    return padded_input, input_length

def eval_kws_dense(full_all_utt_seg_score_dict, vocab, keyword_counts, label_dict, target_dur_dict, full_all_utt_segment_dur_dict, analyze=False):
    # Copied from https://github.com/kamperh/recipe_semantic_flickraudio/blob/master/speech_nn/eval_keyword_spotting.py
    # Keyword spotting evaluation
    
    keywords = sorted(keyword_counts)
    utterances = sorted(full_all_utt_seg_score_dict)
    keyword_ids = [vocab[w] for w in keywords]
    # print(utterances[0])

    # Get sigmoid matrix for keywords
    proposed_min_seg_dict = {}
    keyword_sigmoid_mat = [[] for i in range(len(keyword_ids))]
    for utt in utterances:

        all_utt_seg_score = full_all_utt_seg_score_dict[utt]
        min_scores = np.min(all_utt_seg_score, 0)
        min_segment = np.argmin(all_utt_seg_score, 0)
        proposed_min_seg_dict[utt] = min_segment
        for i, score in enumerate(min_scores):
            keyword_sigmoid_mat[i].append(score)
    keyword_sigmoid_mat = np.array(keyword_sigmoid_mat).T

    # P@10/n for Keyword spotting
    p_at_10 = []
    p_at_n = []
    eer = []

    # P@10/n for Keyword spotting localisation
    p_at_10_loc = []
    p_at_n_loc = []

    if analyze:
        print
    for i_keyword, keyword in enumerate(keywords):       
        # Rank
        rank_order = keyword_sigmoid_mat[:, i_keyword].argsort()[::-1]
        utt_order = [utterances[i] for i in rank_order]
        # ordered_utt_to_id = get_index(samples, utt_order)

        y_true = []
        y_true_loc = []
        for utt in utt_order:
            token_dur_dict = get_token_dur_dict(target_dur_dict[utt])
            if keyword in label_dict[utt]:
                y_true.append(1)
                all_utt_segment = full_all_utt_segment_dur_dict[utt]
                proposed_min_seg = proposed_min_seg_dict[utt]
                proposed_dur = all_utt_segment[proposed_min_seg][vocab[keyword]]
                start_end = token_dur_dict[keyword]
                if (start_end[0] <= proposed_dur[0] < start_end[1] or start_end[0] < proposed_dur[1] <= start_end[1]):
                    y_true_loc.append(1)
                else:
                    y_true_loc.append(0)
            
            else:
                y_true_loc.append(0)
                y_true.append(0)
        y_score = keyword_sigmoid_mat[:, i_keyword][rank_order]
     
        # EER
        cur_eer = calculate_eer(y_true, y_score)
        eer.append(cur_eer)

        # P@10 for keyword spotting
        cur_p_at_10 = float(sum(y_true[:10])) / 10.
        p_at_10.append(cur_p_at_10)

        # P@N for keyword spotting
        cur_p_at_n = float(sum(y_true[:sum(y_true)])) / sum(y_true)
        p_at_n.append(cur_p_at_n)

        # P@10 for keyword spotting localisation
        cur_p_at_10_loc = float(sum(y_true_loc[:10])) / 10.
        p_at_10_loc.append(cur_p_at_10_loc)

        # P@N for keyword spotting for localisation
        if sum(y_true_loc) == 0:
            continue
        cur_p_at_n_loc = float(sum(y_true_loc[:sum(y_true_loc)])) / sum(y_true_loc)
        p_at_n_loc.append(cur_p_at_n_loc)

        if analyze:
            print("-"*79)
            print("Keyword:", keyword)
            print("Keyword spotting")
            print("Current P@10: {:.4f}".format(cur_p_at_10))   
            print("Current P@N: {:.4f}".format(cur_p_at_n))
            print("Current EER: {:.4f}".format(cur_eer))
            print("Keyword spotting localisation")
            print("Current P@10: {:.4f}".format(cur_p_at_10_loc))   
            print("Current P@N: {:.4f}".format(cur_p_at_n_loc))
           
    if analyze:
        print("-"*79)
        print
    # Average Keyword spotting
    p_at_10 = np.mean(p_at_10)
    p_at_n = np.mean(p_at_n)
    eer = np.mean(eer)

    # Average Keyword spotting localisation
    p_at_10_loc = np.mean(p_at_10_loc)
    p_at_n_loc = np.mean(p_at_n_loc)

    return p_at_10, p_at_n, eer, p_at_10_loc, p_at_n_loc       

def calculate_eer(y_true, y_score):
    # https://yangcha.github.io/EER-ROC/
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer
if __name__ == "__main__":
    args = parse_args()

    logger = get_logger()

    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    if args.target_type == "bow":
        VOCAB = data["VOCAB"]
        
    elif args.target_type == "soft":
        VOCAB = data["VOCAB_soft"]
    else:
        print("Invalid target type")

    token_counts = data["word_counts"]
    tokens = list(VOCAB.keys())
    keyword_counts = dict([(i, token_counts[i]) for i in token_counts if i in VOCAB])

    
    samples = data["test"] # change to "test" later on
    
    num_samples = len(samples)
    iVOCAB = dict([(i[1], i[0]) for i in VOCAB.items()])

    l_n_tp = 0
    l_n_fp = 0
    l_n_fn = 0
    score = 0
    total = 0
    

    label_dict = {}
    target_dur_dict = {}
    
    for i in tqdm(range(num_samples)):
        sample = samples[i]
        wave = sample["wave"]
        key = os.path.basename(wave).split(".")[0]
        # print(key)
        gt_trn = [i for i in sample["trn"] if i in VOCAB]
        target_dur = sample["dur"]

        label_dict[key] = gt_trn
        target_dur_dict[key] = target_dur

    full_all_utt_seg_score_dict_fn = "outputs/{}/full_all_utt_seg_score_dict.npz".format(args.model)
    full_all_utt_segment_dur_dict_fn = "outputs/{}/full_all_utt_segment_dur_dict.npz".format(args.model)
    full_all_utt_seg_score_dict = np.load(full_all_utt_seg_score_dict_fn)
    full_all_utt_segment_dur_dict = np.load(full_all_utt_segment_dur_dict_fn)

    p_at_10, p_at_n, eer,  p_at_10_loc, p_at_n_loc = eval_kws_dense(
        full_all_utt_seg_score_dict, VOCAB, keyword_counts, label_dict, target_dur_dict, full_all_utt_segment_dur_dict, args.analyze
        )
    

    print
    print("-"*79)
    print("Keyword spotting")
    print("Average P@10: {:.4f}".format(p_at_10))
    print("Average P@N: {:.4f}".format(p_at_n))
    print("Average EER: {:.4f}".format(eer))
    print("Keyword spotting localisation")
    print("Average P@10: {:.4f}".format(p_at_10_loc))
    print("Average P@N: {:.4f}".format(p_at_n_loc))
    print("-"*79)


# python test.py --model_path 1620808344_psc_bow --target_type bow --test_threshold 0.4
    