import random
import numpy as np
import torch
import glob
import os
from tqdm import tqdm
import argparse
from utils import ensure_folder, eval_detection_prf, extract_feature_test, get_gt_token_duration, get_logger, get_detection_metric_count, eval_localisation_accuracy, get_localisation_metric_count, eval_localisation_prf, get_target_duration, plot_audio
from os import path
from config import pickle_file, device, trained_model_dir, TextGrid_folder, eng_yor_word_file
import pickle
from utils import parse_args, get_yor_eng_word_dict, get_eng_yor_word_dict, plot_location


# Prepare a list of paths to all alignments TextGrids generated using Praat
textgrid_paths_lst = glob.glob(path.join(TextGrid_folder, "*"))
textgrid_base_lst = [os.path.splitext(os.path.basename(file))[0] for file in textgrid_paths_lst]
# print(textgrid_base_lst)
# Get a dictionary containing a mapping from Yoruba word to a list of corresponding English words
yor_to_eng_word_dict = get_yor_eng_word_dict(eng_yor_word_file)
# print(yor_to_eng_word_dict)
# Get a dictionary containing a mapping from English word to Yoruba word
eng_to_yor_word_dict = get_eng_yor_word_dict(eng_yor_word_file)
# print(eng_to_yor_word_dict)


def parse_args():
    parser = argparse.ArgumentParser(description='Keyword detection and localisation in speech')
    parser.add_argument('--model_path', type=str, help='path where the model to be tested is stored')
    parser.add_argument('--target_type', type=str, help='provide the type of target to use for supervision')
    parser.add_argument('--test_threshold', type=float, help='threshold to use during testing')
    parser.add_argument('--seed', type=int)
    parser.add_argument("--plot", action="store_true", help='plotting')
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 
    samples = data["test"] # change to "test" later on

    # print(VOCAB)
    checkpoint =path.join(trained_model_dir, args.model_path, "BEST_checkpoint.tar")
    checkpoint = torch.load(checkpoint, map_location="cpu")
    model = checkpoint["model"].to(device)
    model.eval()
    num_samples = len(samples)
    iVOCAB = dict([(i[1], i[0]) for i in VOCAB.items()])

    d_n_tp = 0
    d_n_tp_fp = 0
    d_n_tp_fn = 0

    l_n_tp = 0
    l_n_fp = 0
    l_n_fn = 0
    score = 0
    total = 0
    all_full_sigmoid_out = {}
    all_attention_weight = {}
    for i in tqdm(range(num_samples)):
        sample = samples[i] 
        wave = sample["wave"]
        key = os.path.basename(wave).split(".")[0]
        # print(key)
        gt_trn = [i for i in sample["trn"] if i in VOCAB]
        # target_dur = [(start_end, dur, tok) for (start_end, dur, tok) in sample["dur"] if  tok.casefold() in VOCAB]
        feature = extract_feature_test(input_file=wave, feature='mfcc', dim=13, cmvn=True, delta=True, delta_delta=True)
        # feature = (feature - feature.mean()) / feature.std()
        padded_input, input_length = pad(feature)
        padded_input = torch.from_numpy(padded_input).unsqueeze(0).to(device)
        input_length = torch.tensor([input_length]).to(device)
        # print(input_length)

        sigmoid_out = torch.from_numpy(10**(np.random.uniform(-3, 0, 67)))
        attention_weights = torch.from_numpy(10**(np.random.uniform(-11, 0, (1, 67, 800))))
        all_full_sigmoid_out[key] = sigmoid_out.squeeze(0).cpu()
        # print("sigmoid shape: ", sigmoid_out.shape)
        # Evaluating model's performance on detection of keywords in one utterance
        hyp_trn = [iVOCAB[i] for i in np.where(sigmoid_out.squeeze(0).cpu() >= args.test_threshold)[0]]
        # print("GT: {}\n HYP: {}".format(gt_trn, hyp_trn))

        d_analysis = get_detection_metric_count(hyp_trn, gt_trn)
        d_n_tp += d_analysis[0]
        d_n_tp_fp += d_analysis[1]
        d_n_tp_fn += d_analysis[2]

        # Evaluating model's performance on localisation of keywords in one utterance
        
        attention_weights = attention_weights.squeeze(0)[:, :input_length]
        all_attention_weight[key] = attention_weights.cpu().numpy()
        # print(attention_weights.shape)
        # print("Attention weights score shape: ", attention_weights.shape)
        tokens = list(VOCAB.keys())
        valid_hyp_trn = [(tok.casefold(), VOCAB[tok.casefold()]) for tok in tokens if tok.casefold() in hyp_trn] # List of words detected by model with a prob > a threshold
        valid_gt_trn = [(tok.casefold(), VOCAB[tok.casefold()]) for tok in gt_trn] # if tok.casefold() in tokens] # remove tokens that are not in the speech vocabulary

        hyp_duration = []
        
        if key not in textgrid_base_lst:
            continue
        ensure_folder("plots/" + key)
        for tok in valid_hyp_trn:
            token_attn_weight = attention_weights.cpu().numpy()[tok[1]]

            if not np.any(token_attn_weight):
                continue
            token_max_frame = np.argmax(token_attn_weight)
            hyp_duration.append((token_max_frame, tok[0]))

            if args.plot:
                import matplotlib.pyplot as plt
                target_dur_yor = get_target_duration(key, textgrid_base_lst, TextGrid_folder)
                fig, axs = plt.subplots(2, 1, figsize=(55, 10))
                plot_audio(axs[0], wave)
                plot_location(axs[1],token_attn_weight, tok, target_dur_yor, key)
                plt.tight_layout()
                file_path = path.join("plots", key, tok[0] + ".pdf")
                plt.savefig(file_path, dpi=150)
                plt.close("all")

        # ground truth start and end time for each word in utterance
        token_gt_duration = get_gt_token_duration(key, textgrid_base_lst, yor_to_eng_word_dict, valid_gt_trn, root_path=TextGrid_folder)

        l_analysis = get_localisation_metric_count(hyp_duration, token_gt_duration)
        l_n_tp += l_analysis[0]
        l_n_fp += l_analysis[1]
        l_n_fn += l_analysis[2]

        s, t = eval_localisation_accuracy(hyp_duration, token_gt_duration)
        score += s
        total += t

    # Compute precision, recall and fscore for detection task
    d_precision, d_recall, d_fscore = eval_detection_prf(d_n_tp, d_n_tp_fp, d_n_tp_fn)

    # Compute precision, recall and fscore for localisation task
    l_precision, l_recall, l_fscore = eval_localisation_prf(l_n_tp, l_n_fp, l_n_fn)
    out_dir = "outputs/" + args.model_path
    ensure_folder(out_dir)
    np.savez_compressed(out_dir + "/all_full_sigmoid_out.npz", **all_full_sigmoid_out)
    np.savez_compressed(out_dir + "/all_attention_weight.npz", **all_attention_weight)
    # Print status
    print
    print("-"*79)
    print("DETECTION SCORES: ")
    print("Sigmoid threshold: {:.2f}".format(args.test_threshold))
    print("No. predictions:", d_n_tp_fp)
    print("No. true tokens:", d_n_tp_fn)
    print("Precision: {} / {} = {:.4f}%".format(d_n_tp, d_n_tp_fp, d_precision*100.))
    print("Recall: {} / {} = {:.4f}%".format(d_n_tp, d_n_tp_fn, d_recall*100.))
    print("F-score: {:.4f}%".format(d_fscore*100.))
    print("-"*79)

    print
    print("-"*79)
    print("LOCALISATION SCORES: ")
    print("Sigmoid threshold: {:.2f}".format(args.test_threshold))
    print("No. predictions:", l_n_fp)
    print("No. true tokens:", l_n_fn)
    print("Precision: {} / {} = {:.4f}%".format(l_n_tp, (l_n_tp + l_n_fp), l_precision*100.))
    print("Recall: {} / {} = {:.4f}%".format(l_n_tp, (l_n_tp + l_n_fn), l_recall*100.))
    print("F-score: {:.4f}%".format(l_fscore*100.))
    print("Accuracy: {} / {} =  {:.4f}%".format(score, total, (score/total) * 100.0))
    print("-"*79)


    
