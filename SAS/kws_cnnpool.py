
import sklearn.metrics as metrics
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import numpy as np
import torch
from tqdm import tqdm
import argparse

import sklearn.metrics as metrics

from utils import extract_feature, get_logger
from os import path
# from models.psc import PSC

from config import pickle_file, device, trained_model_dir, keywords_8_fn
import pickle
from utils import parse_args

def parse_args():
    parser = argparse.ArgumentParser(description='Keyword detection and localisation in speech')
    parser.add_argument('--model_path', type=str, help='path where the model to be tested is stored')
    parser.add_argument('--target_type', type=str, help='provide the type of target to use for supervision')
    parser.add_argument('--test_threshold', type=float, help='threshold to use during testing')
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

def eval_kws(sigmoid_dict, vocab, keyword_counts, utterances, label_dict, analyze=False):
    # Copied from https://github.com/kamperh/recipe_semantic_flickraudio/blob/master/speech_nn/eval_keyword_spotting.py
    # Keyword spotting evaluation
    
    keywords = sorted(keyword_counts)
    utterances = sorted(sigmoid_dict)
    keyword_ids = [vocab[w] for w in keywords]

    print("keyword ids: ", keyword_ids)

    # Get sigmoid matrix for keywords
    keyword_sigmoid_mat = np.zeros((len(utterances), len(keywords)))
    for i_utt, utt in enumerate(utterances):
        keyword_sigmoid_mat[i_utt, :] = sigmoid_dict[utt][keyword_ids]

    p_at_10 = []
    p_at_n = []
    eer = []
    if analyze:
        print
    for i_keyword, keyword in enumerate(keywords):       
        # Rank
        rank_order = keyword_sigmoid_mat[:, i_keyword].argsort()[::-1]
        utt_order = [utterances[i] for i in rank_order]
        # ordered_utt_to_id = get_index(samples, utt_order)
        
        # EER
        y_true = []
        for utt in utt_order:
            if keyword in label_dict[utt]:
                y_true.append(1)
            else:
                y_true.append(0)
        y_score = keyword_sigmoid_mat[:, i_keyword][rank_order]
     
        # print(len(y_score))
        cur_eer = calculate_eer(y_true, y_score)
        eer.append(cur_eer)

        # P@10
        cur_p_at_10 = float(sum(y_true[:10])) / 10.
        p_at_10.append(cur_p_at_10)

        # P@N
        cur_p_at_n = float(sum(y_true[:sum(y_true)])) / sum(y_true)
        p_at_n.append(cur_p_at_n)

        if analyze:
            print("-"*79)
            print("Keyword:", keyword)
            print("Current P@10: {:.4f}".format(cur_p_at_10))   
            print("Current P@N: {:.4f}".format(cur_p_at_n))
            print("Current EER: {:.4f}".format(cur_eer))
        # print(y_true)
    if analyze:
        print("-"*79)
        print
    # Average
    p_at_10 = np.mean(p_at_10)
    p_at_n = np.mean(p_at_n)
    eer = np.mean(eer)

    return p_at_10, p_at_n, eer         

def get_index(samples, utt_order):
    utt_to_index = {}
    for count in range(len(samples)):
        sample = samples[count]
        utt = sample["wave"]
        utt_to_index[utt] = utt_order.index(utt)
        
    return utt_to_index

def calculate_eer(y_true, y_score):
    # https://yangcha.github.io/EER-ROC/
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

if __name__ == "__main__":
    args = parse_args()

    logger = get_logger()
    VOCAB = None
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    if args.target_type == "bow":
        VOCAB = data["VOCAB"]
        
    elif args.target_type == "soft":
        VOCAB = data["VOCAB_soft"]
    else:
        print("Invalid target type")
    
    # Read keywords
    # print("Reading:", keywords_8_fn)
    keywords = []
    with open(keywords_8_fn, "r") as f:
        for line in f:
            keywords.append(line.strip())
    # print("Keywords:", keywords)

    token_counts = data["word_counts"]
    tokens = list(VOCAB.keys())
    keyword_counts = dict([(i, token_counts[i]) for i in token_counts if i in keywords])

    # print(keyword_counts)
    samples = data["test"] # change to "test" later on

    checkpoint =path.join(trained_model_dir, args.model_path, "BEST_checkpoint.tar")
    checkpoint = torch.load(checkpoint, map_location="cpu")
    model = checkpoint["model"].to(device)
    model.eval()
    num_samples = len(samples)
    iVOCAB = dict([(i[1], i[0]) for i in VOCAB.items()])

    # Get sigmoid matrix for keywords
    sigmoid_dict = {}
    label_dict = {}
    utterances = []
    for i in tqdm(range(num_samples)):
        sample = samples[i]
        wave = sample["wave"]
        gt_trn = sample["trn"]
        target_dur = sample["dur"]
        feature = extract_feature(input_file=wave, feature='mfcc', dim=13, cmvn=True, delta=True, delta_delta=True)
        padded_input, input_length = pad(feature)
        padded_input = torch.from_numpy(padded_input).unsqueeze(0).to(device)
        input_length = torch.tensor([input_length]).to(device)
        with torch.no_grad():
            out = model(padded_input)
           
            sigmoid_out = torch.sigmoid(out)
        for j in range(sigmoid_out.shape[0]):
            sigmoid_dict[wave] = sigmoid_out.cpu()[j, :]
            label_dict[wave] = gt_trn

        utterances.append(wave)

    print("Evaluating model's performance on keyword spotting in one utterance")
    p_at_10, p_at_n, eer = eval_kws(
        sigmoid_dict, VOCAB, keyword_counts, utterances, label_dict, args.analyze
        )

    print
    print("-"*79)
    print("Average P@10: {:.4f}".format(p_at_10))
    print("Average P@N: {:.4f}".format(p_at_n))
    print("Average EER: {:.4f}".format(eer))
    print("-"*79)


# python kws.py --model_path 1621334318_psc_bow --target_type bow --analyze