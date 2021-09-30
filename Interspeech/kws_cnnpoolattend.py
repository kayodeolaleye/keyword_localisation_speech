import sklearn.metrics as metrics
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import numpy as np
import torch
from tqdm import tqdm
import argparse
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import sklearn.metrics as metrics

from utils import extract_feature, get_logger, get_token_dur_dict
from os import path
# from models.psc import PSC

from config import pickle_file, device, trained_model_dir, keywords_fn
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

def eval_kws(sigmoid_dict, vocab, keyword_counts, label_dict, target_dur_dict, attention_weights_dict, analyze=False):
    # Copied from https://github.com/kamperh/recipe_semantic_flickraudio/blob/master/speech_nn/eval_keyword_spotting.py
    # Keyword spotting evaluation
    
    keywords = sorted(keyword_counts)
    utterances = sorted(sigmoid_dict)
    keyword_ids = [vocab[w] for w in keywords]

    # print("keyword ids: ", keyword_ids)
    # print("keywords: ", keywords)
    # print("keywords_count: ", keyword_counts)

    # Get sigmoid matrix for keywords
    keyword_sigmoid_mat = np.zeros((len(utterances), len(keywords)))
    for i_utt, utt in enumerate(utterances):
        keyword_sigmoid_mat[i_utt, :] = sigmoid_dict[utt][keyword_ids]

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
                token_attn_weight = attention_weights_dict[utt][vocab[keyword], :]
                time = np.linspace(0, token_attn_weight.shape[0], num=token_attn_weight.shape[0])
                fit = interp1d(time, token_attn_weight, axis=0)
                new_time = np.linspace(0, token_attn_weight.shape[0], num=800)
                attn_weight_interpolated = fit(new_time)
                token_attn_weight_unpadded = attn_weight_interpolated[:input_length]
                if not np.any(token_attn_weight_unpadded):
                    continue
                token_max_frame = np.argmax(token_attn_weight_unpadded)
                start_end = token_dur_dict[keyword]
                if (start_end[0] <= token_max_frame < start_end[1] or start_end[0] < token_max_frame <= start_end[1]):
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
           
        # print(y_true)
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

    token_counts = data["word_counts"]
    tokens = list(VOCAB.keys())
    keyword_counts = dict([(i, token_counts[i]) for i in token_counts if i in VOCAB])

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
    attention_weights_dict = {}
    label_dict = {}
    target_dur_dict = {}

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
            out, attention_weights = model(padded_input)
           
            sigmoid_out = torch.sigmoid(out)
        # print("sigmoid shape: ", sigmoid_out.shape)
        # for j in range(sigmoid_out.shape[0]):
        sigmoid_dict[wave] = sigmoid_out.cpu()
        attention_weights_dict[wave] = attention_weights.squeeze(0)[:, :input_length].cpu().numpy()
        label_dict[wave] = gt_trn
        target_dur_dict[wave] = target_dur

        # utterances[wave] = [gt_trn, target_dur]

    print("Evaluating model's performance on keyword spotting in one utterance")
    p_at_10, p_at_n, eer,  p_at_10_loc, p_at_n_loc = eval_kws(
        sigmoid_dict, VOCAB, keyword_counts, label_dict, target_dur_dict, attention_weights_dict, args.analyze
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


# python kws.py --model_path 1623513734_cnnattend_bow --target_type bow --analyze