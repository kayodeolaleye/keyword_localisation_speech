from os import path
import re
from nltk.corpus import stopwords
import pickle
import argparse
import scipy
import torch
import logging
import numpy as np
from config import flickr8k_folder
import librosa


def parse_args():
    parser = argparse.ArgumentParser(description='Keyword detection and localisation in speech')

    # Low Frame Rate (stacking and skipping frames)
    parser.add_argument('--LFR_m', default=4, type=int,
                        help='Low Frame Rate: number of frames to stack')
    parser.add_argument('--LFR_n', default=3, type=int,
                        help='Low Frame Rate: number of frames to skip')

    # general
    # Network architecture
    # TODO: automatically infer input dim
    
    parser.add_argument('--atype', default='dot', type=str,
                        help='Type of attention (Only support Dot Product now)')

    # Training config
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--half_lr', dest='half_lr', default=True, type=bool,
                        help='Halving learning rate when get small improvement')
    parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                        help='Early stop training when halving lr but still get'
                             'small improvement')
    parser.add_argument('--max_norm', default=5, type=float,
                        help='Gradient norm threshold to clip')
    # minibatch
    parser.add_argument('--batch-size', '-b', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--maxlen_in', default=800, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to generate minibatch')
    # optimizer
    parser.add_argument('--optimizer', default='adam', type=str,
                        choices=['sgd', 'adam'],
                        help='Optimizer (support sgd and adam now)')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Init learning rate')
    parser.add_argument('--momentum', default=0.0, type=float,
                        help='Momentum for optimizer')
    parser.add_argument('--l2', default=1e-5, type=float,
                        help='weight decay (L2 penalty)')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--target_type', type=str, help='provide the type of target to use for supervision')
    parser.add_argument('--val_threshold', type=float, help='threshold to use during validation')
    parser.add_argument('--embed_size', default=1024, type=int, help='embedding dimension / dimension of the convolutional feature')
    parser.add_argument('--vocab_size', default=67, type=int, help='Size of speech corpus vocabulary')
    parser.add_argument('--data_size', default=100, type=int, help='Quantity of speech corpus to use during training')

    args = parser.parse_args()
    return args

def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.makedirs(folder)

# [-0.5, 0.5]
def normalize(yt):
    yt_max = np.max(yt)
    yt_min = np.min(yt)
    a = 1.0 / (yt_max - yt_min)
    b = - (yt_max + yt_min) / (2 * (yt_max - yt_min))

    yt = yt * a + b
    return yt
    
def ctm_to_dict(ctm_fn):
    """
    Return a dictionary with a list of (start, dur, word) for each utterance.
    """
    ctm_dict = {}
    with open(ctm_fn, "r") as f:
        for line in f:
            utt, _, start, dur, word = line.strip().split(" ")
            if not utt in ctm_dict:
                ctm_dict[utt] = []
            start = float(start)
            dur = float(dur)
            ctm_dict[utt].append((start, dur, word))
    return ctm_dict

def get_tran_dict(tran_fn):

    print("Reading:", tran_fn)
    tran_dict = {}
    with open(tran_fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            tran_dict[line[0][5:]] = [i for i in line[1:] if "<" not in i and not "'" in i]

    print("Filtering out stop words")
    tran_content_dict = {}
    for utt in tran_dict:
        tran_content_dict[utt] = [i for i in tran_dict[utt] if i not in stopwords.words("english")]

    tran_dict_fn = path.join(flickr8k_folder, "tran_dict.pkl")
    tran_content_dict_fn = path.join(flickr8k_folder, "tran_content_dict.pkl")
    with open(tran_dict_fn, "wb") as f:
        pickle.dump(tran_dict, f, -1)
    with open(tran_content_dict_fn, "wb") as f:
        pickle.dump(tran_content_dict, f, -1)

    return tran_content_dict

def get_keywords(filename):
    word_to_id_dict = {}
    with open(filename) as f:
        count = 0
        for line in f:
            word_to_id_dict[line.strip()] = count
            count += 1
    return word_to_id_dict

def get_soft_tags(viz_tag_fn, output_dir=None):

    # This function takes an externally generated soft probabilities as a label to train the VisionSpeechCNN and VisionSpeechPSC models.
    # The input is a text file containing 1000 bag-of-words (BoW) and their soft probabilities. The soft probabilities are obtained from an
    # an externally trained BoW multi-layer perceptron built on top of a pretrained VGG16 network.
    # See the papers for more details:
    # - H. Kamper, G. Shakhnarovich, and K. Livescu, "Semantic speech retrieval with a visually grounded model of untranscribed speech,
    # "IEEE/ACM Transactions on Audio, Speech and Language Processing, vol. 27, no. 1, pp. 89-98, 2019. [arXiv](https://arxiv.org/abs/1710.01949)
    # H. Kamper, S. Settle, G. Shakhnarovich, and K. Livescu, "Visually grounded learning of keyword prediction from untranscribed speech," 
    # in Proc. Interspeech, 2017. [arXiv](https://arxiv.org/abs/1706.03818)
    flickr8k_tags_dict = {}

    with open(viz_tag_fn) as f:
        for line in f:
            if not line.isspace():
                line = line.strip().split()
                if line[0] == "#":
                    continue
                key = line[0].replace(":", "")
                flickr8k_tags_dict[key] = [i for i in line[1:]]

            
    soft_tags_dict = dict([(key, np.array(list(map(float, value)), np.float32())) for key, value in flickr8k_tags_dict.items() if key != "Tags"])
    tags = dict([(key, value) for key,value in flickr8k_tags_dict.items() if key == "Tags"])
    vocab_dict = dict([(j, i) for i, j in enumerate(tags["Tags"])])

    return soft_tags_dict, vocab_dict

def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def extract_feature(
    input_file, 
    feature="mfcc", 
    sr=16000, 
    preemph_coef=0.97,
    dim=13, 
    cmvn=True, 
    delta=False, 
    delta_delta=False, 
    window_size=25, 
    stride=10, 
    save_feature=None):

    y, sr = librosa.load(input_file, sr=sr)
    yt, _ = librosa.effects.trim(y, top_db=20)
    yt = normalize(yt)
    yt = preemphasis(yt, preemph_coef)
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)
    if feature == "fbank": # log-scaled
        feat = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=dim, n_fft=ws, hop_length=st)

        feat = np.log(feat + 1e-6)
    elif feature == "mfcc":
        feat = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=dim, n_mels=80, n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rms(yt, hop_length=st, frame_length=ws)

    else:
        raise ValueError("Unsupported Acoustic Feature: " + feature)

    feat = [feat]
    if delta:
        feat.append(librosa.feature.delta(feat[0]))
    if delta_delta:
        feat.append(librosa.feature.delta(feat[0], order=2))
    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
    if save_feature is not None:
        tmp = np.swapaxes(feat, 0, 1).astype("float32")
        np.save(save_feature, tmp)
        return len(tmp)
    else:
        return np.swapaxes(feat, 1, 0).astype("float32")

# Tensorboard functions
def write_scalar_to_tb(
            writer,
            # lr,
            epoch,
            train_loss,
            valid_loss,
            valid_precision,
            valid_recall,
            valid_fscore):
            writer.add_scalar('model/train_loss', train_loss, epoch)
            # writer.add_scalar('model/Learning_rate', lr, epoch)
            writer.add_scalar('model/valid_loss', valid_loss, epoch)
            writer.add_scalar('model/valid_precision', valid_precision, epoch)
            writer.add_scalar('model/valid_recall', valid_recall, epoch)
            writer.add_scalar('model/valid_fscore', valid_fscore, epoch)

def write_hist_to_tb(writer, model, epoch):
    for name, weight in model.named_parameters():
        writer.add_histogram(name, weight, epoch)
        writer.add_histogram(f'{name}.grad', weight.grad, epoch)

def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best_loss, precision, is_best_precision, recall, is_best_recall, fscore, is_best_fscore, model_path):
    state = {'epoch': epoch,
    'epochs_since_improvement':epochs_since_improvement,
    'loss': loss,
    'precision': precision,
    'recall': recall,
    'fscore': fscore,
    'model': model,
    'optimizer': optimizer
    }

    filename = path.join(model_path, 'checkpoint.tar')
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best_loss or is_best_precision or is_best_recall or is_best_fscore:
        torch.save(state, path.join(model_path, 'BEST_checkpoint.tar'))

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_gt_token_duration(target_dur, valid_gt_trn):
            
            token_dur = []
            for start_end, dur, tok in target_dur:
                if tok.casefold() not in [valid_tok for valid_tok, _ in valid_gt_trn]:
                    continue
                token_dur.append((start_end, tok.casefold()))
            return token_dur

def get_detection_metric_count(hyp_trn, gt_trn):
    # Get the number of true positive (n_tp), true positive + false positive (n_tp_fp) and true positive + false negative (n_tp_fn) for a one sample on the detection task
    correct_tokens = set([token for token in gt_trn if token in hyp_trn])
    n_tp = len(correct_tokens)
    n_tp_fp = len(hyp_trn)
    n_tp_fn = len(set(gt_trn))

    return n_tp, n_tp_fp, n_tp_fn

def eval_detection_prf(n_tp, n_tp_fp, n_tp_fn):
    precision = n_tp / n_tp_fp
    recall = n_tp / n_tp_fn
    fscore = 2 * precision * recall / (precision + recall)

    return precision, recall, fscore

def get_localisation_metric_count(hyp_loc, gt_loc):
    # Get the number of true positive (n_tp), true positive + false positive (n_tp_fp) and true positive + false negative (n_tp_fn) for a one sample on the localisation task
    n_tp = 0
    n_fp = 0
    n_fn = 0
    for hyp_frame, hyp_token in hyp_loc:
        if hyp_token not in [gt_token for _, gt_token in gt_loc]:
            n_fp += 1

    for gt_start_end_frame, gt_token in gt_loc:
        if gt_token not in [hyp_token for _, hyp_token in hyp_loc]:
            n_fn += 1
            continue
        for hyp_frame, hyp_token in hyp_loc:
            if hyp_token == gt_token and (gt_start_end_frame[0] <= hyp_frame[0] < gt_start_end_frame[1] or gt_start_end_frame[0] <= hyp_frame[1] <= gt_start_end_frame[1]):
                n_tp += 1
            elif hyp_token == gt_token and (hyp_frame[0] < gt_start_end_frame[0] and gt_start_end_frame[1] < hyp_frame[1]):
                n_fp += 1

    return n_tp, n_fp, n_fn

def eval_localisation_prf(n_tp, n_fp, n_fn):
    precision = n_tp / (n_tp + n_fp)
    recall = n_tp / (n_tp + n_fn)
    fscore = 2 * precision * recall / (precision + recall)

    return precision, recall, fscore


def compute_cam(grad_cam, x, iVOCAB):
    cams_dict = {}
    for i in range(1000):
        token = iVOCAB[i]
        cam = grad_cam.generate_cam(x, target_class=i)
        #cam = np.broadcast_to(cam, (39, cam.shape[0]))
        cams_dict[token] = cam
    return cams_dict

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