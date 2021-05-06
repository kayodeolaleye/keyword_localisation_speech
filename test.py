import numpy as np
import torch
from tqdm import tqdm
import argparse
# from pre_process import VOCAB
from utils import eval_prf, extract_feature, get_logger, get_metric_count
from os import path
from models.psc import PSC

from config import pickle_file, device, trained_model_dir
import pickle
from utils import parse_args

def parse_args():
    parser = argparse.ArgumentParser(description='Keyword detection and localisation in speech')
    parser.add_argument('--model_path', type=str, help='path where the model to be tested is stored')
    parser.add_argument('--target_type', type=str, help='provide the type of target to use for supervision')
    parser.add_argument('--test_threshold', type=float, help='threshold to use during testing')

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

    VOCAB = data["VOCAB"]
    if args.target_type == "bow":
        VOCAB = data["VOCAB"]
        
    elif args.target_type == "soft":
        VOCAB = data["VOCAB_soft"]
    else:
        print("Invalid target type")

    
    samples = data["dev"] # change to "test" later on

    # filename = path.join(trained_model_dir, args.model_path,, "psc_" + args.target_type + "_model.pth")
    # model = PSC(args.out_dim, args.temp_ratio)
    # model.load_state_dict(torch.load(filename))
    # model = model.to(device)
    # model.eval()

    checkpoint =path.join(trained_model_dir, args.model_path, "BEST_checkpoint_" + args.target_type + ".tar")
    checkpoint = torch.load(checkpoint, map_location="cpu")
    model = checkpoint["model"].to(device)
    model.eval()
    num_samples = len(samples)
    iVOCAB = dict([(i[1], i[0]) for i in VOCAB.items()])

    n_tp = 0
    n_tp_fp = 0
    n_tp_fn = 0

    for i in tqdm(range(num_samples)):
        sample = samples[i]
        wave = sample["wave"]
        gt_trn = sample["trn"]
        feature = extract_feature(input_file=wave, feature='mfcc', dim=13, cmvn=True, delta=True, delta_delta=True)
        padded_input, input_length = pad(feature)
        padded_input = torch.from_numpy(padded_input).unsqueeze(0).to(device)
        input_length = torch.tensor([input_length]).to(device)
        # print("Input length: ", input_length.shape)
        with torch.no_grad():
            out, frame_score = model(padded_input, input_length)
            sigmoid_out = torch.sigmoid(out)
    
        hyp_trn = [iVOCAB[i] for i in np.where(sigmoid_out.squeeze(0).cpu() >= args.test_threshold)[0]]
        # print("GT: {}\n HYP: {}".format(gt_trn, hyp_trn))

        analysis = get_metric_count(hyp_trn, gt_trn)
        n_tp += analysis[0]
        n_tp_fp += analysis[1]
        n_tp_fn += analysis[2]
    

    # Compute precision, recall and fscore
    precision, recall, fscore = eval_prf(n_tp, n_tp_fp, n_tp_fn)

    # Print status
    print
    print("-"*79)
    print("Sigmoid threshold: {:.2f}".format(args.test_threshold))
    print("No. predictions:", n_tp_fp)
    print("No. true tokens:", n_tp_fn)
    print("Precision: {} / {} = {:.4f}%".format(n_tp, n_tp_fp, precision*100.))
    print("Recall: {} / {} = {:.4f}%".format(n_tp, n_tp_fn, recall*100.))
    print("F-score: {:.4f}%".format(fscore*100.))
    print("-"*79)

# python test.py --target_type bow --test_threshold 0.4
    
