# Prepare data for pre-processing

import os
from os import path
import shutil
import glob
from utils import ctm_to_dict, ensure_folder
from config import flickr8k_trans_dir, flickr8k_wav_dir, wav_to_spk_fn, wav_folder, flickr8k_ctm_fn, tran_folder

def prepare_data_split(base_dir, splits=["train", "dev", "test"]):

    subset_fn = path.join(flickr8k_trans_dir, "Flickr_8k.{}Images.txt".format(subset))
    keys = []
    with open(subset_fn) as f:
        for line in f:
            keys.append(path.splitext(line.strip())[0])

    for subset in splits:
        source = os.path.join(base_dir, "flickr_audio_yoruba_" + subset)
        wav_list = glob.glob(path.join(source, "*"))
        for file in wav_list:
            basename = path.splitext(path.basename(file))[0]
            speaker = basename[:4]
            destination = path.join(wav_folder, subset, speaker)
            ensure_folder(destination)
            if basename[4:-2] in keys:
                old = path.join(source, basename + ".wav")
                new = path.join(destination,  basename[4:] + ".wav")
                os.rename(old, new)


def clean_transcription():
    ctm_dict = ctm_to_dict(flickr8k_ctm_fn)

    utterances = []
    n_missing = 0
    with open(wav_to_spk_fn, "r") as f:
        for line in f:
            wav, speaker = line.strip().split(" ")
            speaker = "{:03d}".format(int(speaker))
            utt = speaker + "_" + path.splitext(wav)[0]
            ctm_label = utt[4:-2] + ".jpg_#" + utt[-1]
            if not ctm_label in ctm_dict:
                n_missing += 1
                continue
            utterances.append(utt)

    ensure_folder(tran_folder)
    tran_eng_fn = path.join(tran_folder, "flickr8k_transcript_eng.txt")
    tran_yor_fn = path.join(tran_folder, "flickr8k_transcript_yor.txt")
    with open(tran_eng_fn, "w") as text_f:
        for utt in sorted(utterances):
            ctm_label = utt[4:-2] + ".jpg_#" + utt[-1]
            ctm_entry = ctm_dict[utt[4:-2] + ".jpg_#" + utt[-1]]
            text_f.write("S"+utt + " " + " ".join([i[2].lower() for i in ctm_entry]) + "\n")

if __name__== "__main__":
    prepare_data_split()
    clean_transcription()



# python structure_data.py