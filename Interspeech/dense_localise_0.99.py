from librosa.core.audio import get_duration
import numpy as np
import torch
from tqdm import tqdm
import argparse
from utils import ensure_folder, extract_feature, get_gt_token_duration, get_logger, get_localisation_metric_count, eval_localisation_prf
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

def split_frame_length(frame_length, min_frame, max_frame, step):

        """
        Args 
        frame_length (int): length of a speech utterance 

        output
        segments_duration (list of tuple): the start and end of each split.
        """

        segments_duration = []
        
        start = 0

        while start < frame_length:
            end = start + min_frame
            while end <= frame_length and end - start <= max_frame:
                segments_duration.append((start, end))
                end += step
            start += step
        return segments_duration

def plot_stuff(valid_proposed_max_durations, all_utt_segment_dur, all_utt_seg_score, target_dur, wave_path, ivocab):
    utt_key = path.splitext(path.basename(wave_path))[0]
    plt.figure(figsize=(20,10), dpi=80)
    
    for word_id, segment_ind in valid_proposed_max_durations:
        dur = all_utt_segment_dur[segment_ind]
        mid_dur = np.sum(dur)/2
        text = str(dur[0]) + " - " + str(dur[1]) + "(" + ivocab[word_id] + ")"
        plt.annotate(text, (mid_dur, all_utt_seg_score[segment_ind, word_id]), fontsize=16)
        plt.plot(mid_dur, all_utt_seg_score[segment_ind, word_id], "r+", markersize=19)

    def timelines(y, xstart, xstop, color='b', label=None):
        """Plot timelines at y from xstart to xstop with given color."""   
        plt.hlines(y, xstart, xstop, color, lw=2)
        plt.vlines(xstart, y+0.03, y-0.03, color, lw=1)
        plt.vlines(xstop, y+0.03, y-0.03, color, lw=1)
        if label is not None:
            plt.text(xstart + (xstop-xstart)/2.0, y+0.03, label, horizontalalignment='center')

    for start_end, dur, tok in target_dur:
        timelines(0.25, start_end[0], start_end[1], "k", tok)
        
    plt.title("Dense", fontsize=26)
    plt.xlabel("Time (frames)")
    plt.xlim(0.0)
    plt.ylim(0.0)
    ensure_folder("plots")
    file_path = path.join("plots", utt_key + ".pdf")
    plt.savefig(file_path, dpi=150)

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
     
        for i, (start, end) in enumerate(segments_dur):
            segment_key = str(start) + "_" + str(end)
            segment = padded_input[:, start:end]
            segment = np.transpose(segment, (1, 0))
            padded_segment, seg_len = pad(segment)
            padded_segment = torch.from_numpy(padded_segment).unsqueeze(0).to(device)
            # Feed padded segment to trained mode
            with torch.no_grad():
                out, __ = model(padded_segment)
                sigmoid_out = torch.sigmoid(out)
                sigmoid_out = sigmoid_out.squeeze(0).cpu().numpy()
         
            all_utt_seg_score[i, :] = sigmoid_out
            all_utt_segment_dur.append((start, end))

        proposed_max_durations = np.argmax(all_utt_seg_score, 0)  # size = 67
        valid_proposed_max_durations = []
        for word_id, segment_ind in enumerate(proposed_max_durations):
            sig_out = all_utt_seg_score[segment_ind]
            # valid_words = [iVOCAB[i] for i in np.where(sig_out >= 0.99)[0]]
            if sig_out[word_id] >= 0.99:
                valid_proposed_max_durations.append((word_id, segment_ind))
        
        hyp_duration = [(np.sum(all_utt_segment_dur[i_segment])/2, iVOCAB[i_word]) for i_word, i_segment in valid_proposed_max_durations]
        # print(hyp_duration)
        valid_gt_trn = [(tok.casefold(), VOCAB[tok.casefold()]) for tok in gt_trn] 
        token_gt_duration = get_gt_token_duration(target_dur, valid_gt_trn) # ground truth start and end time for each word in utterance

        # Evaluating model's performance on localisation of keywords in one utterance
        l_analysis = get_localisation_metric_count(hyp_duration, token_gt_duration)
        l_n_tp += l_analysis[0]
        l_n_fp += l_analysis[1]
        l_n_fn += l_analysis[2]

        plot_stuff(valid_proposed_max_durations, all_utt_segment_dur, all_utt_seg_score, target_dur_full, wave, iVOCAB)
    # # Compute precision, recall and fscore for localisation task
    l_precision, l_recall, l_fscore = eval_localisation_prf(l_n_tp, l_n_fp, l_n_fn)

    print
    print("-"*79)
    print("LOCALISATION SCORES: ")
    print("Sigmoid threshold: {:.2f}".format(0.99))
    print("No. predictions:", l_n_fp + l_n_tp)
    print("No. true tokens:", l_n_fn + l_n_tp)
    print("Precision: {} / {} = {:.4f}%".format(l_n_tp, (l_n_tp + l_n_fp), l_precision*100.))
    print("Recall: {} / {} = {:.4f}%".format(l_n_tp, (l_n_tp + l_n_fn), l_recall*100.))
    print("F-score: {:.4f}%".format(l_fscore*100.))
    print("-"*79)


# python test.py --model_path 1620808344_psc_bow --target_type bow --test_threshold 0.4
    