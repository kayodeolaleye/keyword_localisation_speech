from librosa.core.audio import get_duration
import numpy as np
import torch
from tqdm import tqdm
import argparse
from utils_masked import ensure_folder, extract_feature, get_gt_token_duration, get_logger, get_localisation_metric_count, eval_localisation_accuracy, eval_localisation_prf, split_frame_length, plot_stuff
from os import path
from config import pickle_file, device, trained_model_dir
import pickle
from utils import parse_args
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Keyword detection and localisation in speech')
    parser.add_argument('--model_path', type=str, help='path where the model to be tested is stored')
    parser.add_argument('--target_type', type=str, help='provide the type of target to use for supervision')
    parser.add_argument('--test_threshold', type=float, help='threshold to use during testing')
    parser.add_argument('--min_frame', type=int, help='')
    parser.add_argument('--max_frame', type=int, help='')
    parser.add_argument('--step', type=int, help='')

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

    
    samples = data["test"] # change to "test" later on
    # print(VOCAB)
    checkpoint =path.join(trained_model_dir, args.model_path, "BEST_checkpoint.tar")
    checkpoint = torch.load(checkpoint, map_location="cpu")
    model = checkpoint["model"].to(device)
    model.eval()
    num_samples = len(samples)
    iVOCAB = dict([(i[1], i[0]) for i in VOCAB.items()])

    l_n_tp = 0
    l_n_fp = 0
    l_n_fn = 0

    score = 0
    total = 0

    for i in tqdm(range(num_samples)):
        sample = samples[i]
        wave = sample["wave"]
        gt_trn = [i for i in sample["trn"] if i in VOCAB]
        # print("gt_trn: ", gt_trn)
        target_dur_full = sample["dur"]
        target_dur = [(start_end, dur, tok) for (start_end, dur, tok) in sample["dur"] if  tok.casefold() in VOCAB]
        feature = extract_feature(input_file=wave, feature='mfcc', dim=13, cmvn=True, delta=True, delta_delta=True)
        padded_input, input_length = pad(feature)
        segments_dur = split_frame_length(input_length, args.min_frame, args.max_frame, args.step)
        all_utt_seg_score = np.zeros((len(segments_dur), 67))
        all_utt_segment_dur = []

        with torch.no_grad():
            full_out, attention_weights = model(torch.from_numpy(padded_input).unsqueeze(0).to(device))
            
        full_sigmoid_out = torch.sigmoid(full_out)
     
        for i, (start, end) in enumerate(segments_dur):
            segment_key = str(start) + "_" + str(end)
            # segment = padded_input[:, start:end]
            masked_padded_input = padded_input
            masked_padded_input[:, start:end] = 0
            
            # segment = np.transpose(segment, (1, 0))
            # padded_segment, seg_len = pad(segment)
            masked_padded_input = torch.from_numpy(masked_padded_input).unsqueeze(0).to(device)
            # Feed padded segment to trained mode
            with torch.no_grad():
                out, __ = model(masked_padded_input)
                sigmoid_out = torch.sigmoid(out)
                sigmoid_out = sigmoid_out.squeeze(0).cpu().numpy()
         
            all_utt_seg_score[i, :] = sigmoid_out
            all_utt_segment_dur.append((start, end))

        proposed_min_durations = np.argmin(all_utt_seg_score, 0)  # size = 67
        valid_proposed_min_durations = []
        for word_id, segment_ind in enumerate(proposed_min_durations):
            sig_out = all_utt_seg_score[segment_ind]
            if full_sigmoid_out[word_id] >= args.test_threshold:
                valid_proposed_min_durations.append((word_id, segment_ind))
        
        hyp_duration = [(all_utt_segment_dur[i_segment], iVOCAB[i_word]) for i_word, i_segment in valid_proposed_min_durations]
        # print(hyp_duration)
        valid_gt_trn = [(tok.casefold(), VOCAB[tok.casefold()]) for tok in gt_trn] 
        token_gt_duration = get_gt_token_duration(target_dur, valid_gt_trn) # ground truth start and end time for each word in utterance

        
        # Evaluating model's performance on localisation of keywords in one utterance
        l_analysis = get_localisation_metric_count(hyp_duration, token_gt_duration)
        l_n_tp += l_analysis[0]
        l_n_fp += l_analysis[1]
        l_n_fn += l_analysis[2]

        s, t = eval_localisation_accuracy(hyp_duration, token_gt_duration)
        score += s
        total += t

        # plot_stuff(valid_proposed_min_durations, all_utt_segment_dur, all_utt_seg_score, target_dur_full, wave, iVOCAB)
    
    # # Compute precision, recall and fscore for localisation task
    l_precision, l_recall, l_fscore = eval_localisation_prf(l_n_tp, l_n_fp, l_n_fn)

    print
    print("-"*79)
    print("LOCALISATION SCORES: ")
    print("Sigmoid threshold: {:.2f}".format(args.test_threshold))
    print("Precision: {} / {} = {:.4f}%".format(l_n_tp, (l_n_tp + l_n_fp), l_precision*100.))
    print("Recall: {} / {} = {:.4f}%".format(l_n_tp, (l_n_tp + l_n_fn), l_recall*100.))
    print("F-score: {:.4f}%".format(l_fscore*100.))
    print("Accuracy: {} / {} =  {:.4f}%".format(score, total, (score/total) * 100.0))
    print("-"*79)


# python test.py --model_path 1620808344_psc_bow --target_type bow --test_threshold 0.4
    