
from tqdm import tqdm
import pickle
from os import path
from os import listdir
from utils import get_tran_dict, get_soft_tags, ctm_to_dict
from config import tran_folder, wav_folder, pickle_file, soft_tags_fn, flickr8k_ctm_fn
from collections import Counter


def get_data(split):

    print("Getting {} data...".format(split))
    tran_fn = path.join(tran_folder, "flickr8k_transcript.txt")
    tran_dict = get_tran_dict(tran_fn)
    # word_tokens = []
    # for utt in tran_dict:
    #     word_tokens.extend(tran_dict[utt])
    
    # Get soft labels dictionary from an external visual tagger for 1000 keywords
    soft_tags_dict, _ = get_soft_tags(soft_tags_fn)

    # Forced alignments (ctm) 
    ctm_dict = ctm_to_dict(flickr8k_ctm_fn)

    samples = []
    folder = path.join(wav_folder, split)
    dirs = [path.join(folder, d) for d in listdir(folder) if path.isdir(path.join(folder, d))]
    for dir in tqdm(dirs):
        files = [f for f in listdir(dir) if f.endswith('.wav')]

        for f in files:
            target_dur = []
            wave = path.join(dir, f)
            key = f.split('.')[0]
            if key in tran_dict:
                trn = tran_dict[key]
                soft = soft_tags_dict[key[:-2]]

                ctm_entry = ctm_dict[key[:-2] + ".jpg_#" + key[-1]]
                utt_start = ctm_entry[0][0]
                for start, dur, label in ctm_entry:
                    xstart = round((start - utt_start) * 100)
                    xstop = round((start - utt_start + dur) * 100)
                    target_dur.append(((xstart, xstop), dur * 100, label))

                samples.append({'trn': trn, 'soft': soft, 'wave': wave, 'dur': target_dur})
    print("Split: {}, num_files: {}".format(split, len(samples)))
    
    return samples

def get_vocab():

    tran_fn = path.join(tran_folder, "flickr8k_transcript.txt")
    tran_dict = get_tran_dict(tran_fn)
    word_tokens = []
    for utt in tran_dict:
        word_tokens.extend(tran_dict[utt])
    VOCAB = dict([
        (j[0], i) for i, j in enumerate(Counter(word_tokens).most_common(1000))
        ])

    VOCAB_fn = path.join(tran_folder, "VOCAB.pkl")
    
    with open(VOCAB_fn, "wb") as f:
        pickle.dump(VOCAB, f, -1)

    # Get soft labels dictionary from an external visual tagger for 1000 keywords
    _, VOCAB_soft = get_soft_tags(soft_tags_fn)

    VOCAB_soft_fn = path.join(tran_folder, "VOCAB_soft.pkl")
    with open(VOCAB_soft_fn, "wb") as f:
        pickle.dump(VOCAB_soft, f, -1)    

    word_counts = Counter(word_tokens)

    return VOCAB, VOCAB_soft, word_counts

if __name__ == "__main__":
    
    VOCAB, VOCAB_soft, word_counts = get_vocab()
    data = dict()
    data["word_counts"] = word_counts
    data["VOCAB"] = VOCAB
    data["VOCAB_soft"] = VOCAB_soft
    data["train"] = get_data("train")
    data["dev"] = get_data("dev")
    data["test"] = get_data("test")

    with open(pickle_file, "wb") as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_dev: ' + str(len(data['dev'])))
    print('num_test: ' + str(len(data['test'])))
    print('vocab_size: ' + str(len(data['VOCAB'])))

# python pre_process.py