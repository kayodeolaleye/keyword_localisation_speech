# Prepare data for pre-processing
from email.mime import base
import os
from os import path
import glob
import shutil
from utils import ctm_to_dict, ensure_folder
from config import wav_to_spk_fn, wav_folder, flickr8k_ctm_fn, tran_folder, path_to_yoruba_utt, flickr8k_wav_dir
print(path_to_yoruba_utt)
def prepare_data_split(splits=["train", "dev", "test"]):
    for subset in splits:
        print("Preparing {} data for pre-processing".format(subset))
        yoruba_source = os.path.join(path_to_yoruba_utt, subset, "S001")
        wav_list = glob.glob(path.join(yoruba_source, "*"))
        destination = path.join(wav_folder, subset, "S001")
        ensure_folder(destination)
        for utt in wav_list:
            basename = os.path.splitext(os.path.basename(utt))[0]
            english_source = path.join(flickr8k_wav_dir, basename + ".wav")
            dest = shutil.copy(english_source, destination)

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
    tran_fn = path.join(tran_folder, "flickr8k_transcript.txt")
    with open(tran_fn, "w") as text_f:
        for utt in sorted(utterances):
            ctm_label = utt[4:-2] + ".jpg_#" + utt[-1]
            ctm_entry = ctm_dict[utt[4:-2] + ".jpg_#" + utt[-1]]
            text_f.write("S"+utt + " " + " ".join([i[2].lower() for i in ctm_entry]) + "\n")

if __name__== "__main__":
    prepare_data_split()
    clean_transcription()
