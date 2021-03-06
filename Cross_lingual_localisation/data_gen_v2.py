import pickle
import torch
import random
from os import path
from utils import extract_feature_train, parse_args, spec_augment
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.dataloader import default_collate
from config import pickle_file, input_dim, num_workers, tran_folder, soft_tags_fn

class Flickr8kDataset(Dataset):
    def __init__(self, subset):
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
        feature = extract_feature_train(input_file=wave, feature="mfcc", dim=13, cmvn=True, delta=True, delta_delta=True)
    
        # Zero mean and unit variance
        feature = (feature - feature.mean()) / feature.std()
        
        feature = spec_augment(feature)
        
        return feature, trn, soft, dur

    def __len__(self):
        return len(self.samples)

def get_bow_vector(tran):
    vocab_fn = path.join(tran_folder, "VOCAB.pkl")
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

    batch_len = len(batch)
    # print(batch_len)
    
    padded_input_batch = torch.tensor(np.zeros((batch_len, 800, 39), dtype=np.float32).transpose(0,2,1))
    # print(padded_input_batch.shape)
    
    bow_vector_batch = torch.from_numpy(np.array([get_bow_vector(x[1]) for x in batch]))
    # print(len(bow_vector_batch))

    soft_batch = torch.from_numpy(np.array([x[2] for x in batch]))
    # print(len(soft_batch))

    dur_batch = [x[3] for x in batch]

    input_length_batch = torch.from_numpy(np.repeat([800], batch_len))
    # print(input_length_batch)
    # print(input_length_batch)


    # for i, elem in enumerate(batch):
    #     feature, trn, soft, dur = elem
    

    #     bow_vector = get_bow_vector(trn)
        
        # print("------------------------------")
        # print((np.transpose(padded_input, (1, 0)), bow_vector, soft, dur, input_length))
        # print(padded_input.shape, bow_vector.shape, len(soft), len(dur), input_length)
        # print("------------------------------")
        # batch[i] = (np.transpose(padded_input, (1, 0)), bow_vector, soft, dur, input_length)

    # sort it by input lengths (long to short)
    # batch.sort(key=lambda x: x[4], reverse=True)

    return padded_input_batch, bow_vector_batch, soft_batch, dur_batch, input_length_batch

# if __name__ == "__main__":
#     args = parse_args()
#     train_dataset = Flickr8kDataset('train')
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=num_workers,
#                                                pin_memory=True, collate_fn=pad_collate)

#     print('len(train_dataset): ' + str(len(train_dataset)))
#     print('len(train_loader):' + str(len(train_loader)))

#     feature = train_dataset[10][0]
#     print('feature.shape: ' + str(feature.shape))

#     trn = train_dataset[10][1]
#     print('trn: ' + str(trn))

#     soft = train_dataset[10][2]
#     print('soft: ', str(soft.shape))

#     dur = train_dataset[10][3]
#     print('dur: ', str(dur))

#     for data in train_loader:
#         padded_input, bow_target, soft_target, target_dur, input_lengths = data
#         print('padded_input: ' + str(padded_input))
#         print('bow_target: ' + str(bow_target))
#         print('soft_target: ' + str(soft_target))
#         print('target_duration: ' + str(target_dur))
#         print('input_lengths: ' + str(input_lengths))
#         print("Shape of utt: ", padded_input.shape)
#         print("Shape of bow target: ", bow_target.shape)
#         print("Shape of soft target: ", soft_target.shape)
#         break

