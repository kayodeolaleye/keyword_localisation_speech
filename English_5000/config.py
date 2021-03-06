from os import path
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
input_dim = 39  # dimension of feature

num_workers = 0  # for data-loading; right now, only 1 works with h5py

# Training parameters
print_freq = 100  # print training/validation stats  every __ batches

DATA_DIR = 'data'
flickr8k_folder = 'data/flickr8k'
tran_folder = path.join(flickr8k_folder, "transcript")
wav_folder = path.join(flickr8k_folder, 'wav')
pickle_file = 'data/flickr8k.pickle'

path_to_yoruba_utt = "/home/kayode/PhD/Journal_stuff/Clean_recipes_localisation/Cross_lingual_localisation/data/flickr8k/wav"
# Raw data path
BASE_DIR = "/home/kayode/data"
flickr8k_trans_dir = path.join(BASE_DIR, "Flickr8k_text")
flickr8k_audio_dir = path.join(BASE_DIR, "flickr_audio")
flickr8k_wav_dir = path.join(flickr8k_audio_dir, "wavs")
wav_to_spk_fn = path.join(flickr8k_audio_dir, "wav2spk.txt")
flickr8k_ctm_fn = path.join(flickr8k_audio_dir, "flickr_8k.ctm")
soft_tags_fn = path.join(flickr8k_folder, "flickr8k.tags.all.txt") # path to Herman's soft tags
keywords_fn = path.join(flickr8k_folder, "keywords.8.txt")

# Model path
trained_model_dir = "trained_models"
