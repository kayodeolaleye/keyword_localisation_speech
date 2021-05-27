from inspect import ArgInfo
import pickle
import random
import torch
from os import path
from utils import extract_feature, parse_args, extract_mfcc
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.dataloader import default_collate
from config import pickle_file, num_workers, tran_folder

class Flickr8kDataset(Dataset):
    def __init__(self, subset, target_type):
        self.target_type = target_type
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)
        # self.vocab = data['VOCAB']
        self.samples = data[subset]
        # self.args = args
        print("Loading {} {} samples...".format(len(self.samples), subset))

    def __getitem__(self, i):
        sample = self.samples[i]
        wave = sample["wave"]
        trn = sample["trn"]
        soft = sample["soft"]
        dur = sample["dur"]
        bow = get_bow_vector(trn, self.target_type)
        feature = extract_feature(input_file=wave, feature="mfcc", dim=13, cmvn=True, delta=True, delta_delta=True)
        # feature = extract_mfcc(input_file=wave)
        # feature = build_LFR_features(feature, m=self.args.LFR_m, n = self.args.LFR_n)
        # Zero mean and unit variance
        feature = (feature - feature.mean()) / feature.std()
        
        feature = spec_augment(feature)
        
        return feature, bow, soft, dur

    def __len__(self):
        return len(self.samples)

def get_bow_vector(tran, target_type):
    vocab_fn = None
    if target_type == "bow":
        vocab_fn = path.join(tran_folder, "VOCAB.pkl")
    elif target_type == "soft":
         vocab_fn = path.join(tran_folder, "VOCAB_soft.pkl")
    else:
        print("Invalid target type")
    with open(vocab_fn, "rb") as f:
        vocab = pickle.load(f)
    bow_vector =list( np.zeros((len(vocab))))
    for word in tran:
        if word in vocab:
            bow_vector[vocab[word]] = 1
    return np.array(bow_vector, np.float32)

def pad_collate(batch):
    max_input_len = 800

    # for elem in batch:
    #     feature, trn, soft = elem
    #     max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
        # max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

    for i, elem in enumerate(batch):
        feature, bow, soft, dur = elem
        input_length = feature.shape[0]
        input_dim = feature.shape[1]
        padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
        length = min(input_length, max_input_len)
        padded_input[:length, :] = feature[:length, :]

        # bow_vector = get_bow_vector(trn)
    
        batch[i] = (np.transpose(padded_input, (1, 0)), bow, soft, dur, input_length)

    # sort it by input lengths (long to short)
    batch.sort(key=lambda x: x[4], reverse=True)

    return default_collate(batch)


def spec_augment(spec: np.ndarray,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20,
                 value=0):
    spec = spec.copy()
    num_mask = random.randint(1, num_mask)
    for i in range(num_mask):
        all_freqs_num, all_frames_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[f0:f0 + num_freqs_to_mask, :] = value

        time_percentage = random.uniform(0.0, time_masking)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[:, t0:t0 + num_frames_to_mask] = value
    return spec


if __name__ == "__main__":
    args = parse_args()
    train_dataset = Flickr8kDataset('train', args.target_type)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=num_workers,
                                               pin_memory=True, collate_fn=pad_collate)

    print('len(train_dataset): ' + str(len(train_dataset)))
    print('len(train_loader):' + str(len(train_loader)))

    feature = train_dataset[10][0]
    print('feature.shape: ' + str(feature.shape))

    trn = train_dataset[10][1]
    print('trn: ' + str(trn))

    soft = train_dataset[10][2]
    print('soft: ', str(soft.shape))

    dur = train_dataset[10][3]
    print('dur: ', str(dur))

    for data in train_loader:
        padded_input, bow_target, soft_target, target_dur, input_lengths = data
        print('padded_input: ' + str(padded_input))
        print('bow_target: ' + str(bow_target))
        print('soft_target: ' + str(soft_target))
        print('target_duration: ' + str(target_dur))
        print('input_lengths: ' + str(input_lengths))
        print("Shape of utt: ", padded_input.shape)
        print("Shape of bow target: ", bow_target.shape)
        print("Shape of soft target: ", soft_target.shape)
        break

