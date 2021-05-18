import numpy as np
import torch
from tqdm import tqdm
import argparse
# from pre_process import VOCAB
from utils import compute_cam, eval_detection_prf, eval_localisation_prf, extract_feature, get_gt_token_duration, get_localisation_metric_count, get_logger, get_detection_metric_count
from os import path
# from models.cnn_pool import CNNPool
from models.gradcam import GradCAM
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from config import pickle_file, device, trained_model_dir
import pickle
from utils import parse_args

def parse_args():
    parser = argparse.ArgumentParser(description='Keyword detection and localisation in speech')
    parser.add_argument('--model_path', type=str, help='path where the model to be tested is stored')
    parser.add_argument('--target_type', type=str, help='provide the type of target to use for supervision')
    parser.add_argument('--test_threshold', type=float, help='threshold to use during testing')
    parser.add_argument('--target_layer', type=int, default=7, help="Layer to compute gradcam on")

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
    VOCAB = None
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
    
    checkpoint =path.join(trained_model_dir, args.model_path, "BEST_checkpoint.tar")
    checkpoint = torch.load(checkpoint, map_location="cpu")
    model = checkpoint["model"].to(device)
    
    grad_cam = GradCAM(model, target_layer=args.target_layer) # Create an object for GradCAM computation
 
    model.eval()
    num_samples = len(samples)
    iVOCAB = dict([(i[1], i[0]) for i in VOCAB.items()])

    d_n_tp = 0
    d_n_tp_fp = 0
    d_n_tp_fn = 0

    l_n_tp = 0
    l_n_fp = 0
    l_n_fn = 0

    for i in tqdm(range(num_samples)):
        sample = samples[i]
        wave = sample["wave"]
        gt_trn = sample["trn"]
        target_dur = sample["dur"]
        feature = extract_feature(input_file=wave, feature='mfcc', dim=13, cmvn=True, delta=True, delta_delta=True)
        padded_input, input_length = pad(feature)
        padded_input = torch.from_numpy(padded_input).unsqueeze(0).to(device)
        # print("Input length: ", input_length.shape)
        with torch.no_grad():
            out = model(padded_input)
            sigmoid_out = torch.sigmoid(out)
    
        # Evaluating model's performance on detection of keywords in one utterance
        hyp_trn = [iVOCAB[i] for i in np.where(sigmoid_out.squeeze(0).cpu() >= args.test_threshold)[0]]
        # print("GT: {}\n HYP: {}".format(gt_trn, hyp_trn))

        d_analysis = get_detection_metric_count(hyp_trn, gt_trn)
        d_n_tp += d_analysis[0]
        d_n_tp_fp += d_analysis[1]
        d_n_tp_fn += d_analysis[2]

        
        # GradCAM computation for a single utterance.
        cams_dict = compute_cam(grad_cam, padded_input, iVOCAB)

        # Evaluating model's performance on localisation of keywords in one utterance
        tokens = list(VOCAB.keys())
        valid_hyp_trn = [(tok.casefold(), VOCAB[tok.casefold()]) for tok in tokens if tok.casefold() in hyp_trn] # List of words detected by model with a prob > a threshold
        valid_gt_trn = [(tok.casefold(), VOCAB[tok.casefold()]) for tok in gt_trn if tok.casefold() in tokens] # remove tokens that are not in the speech vocabulary

        token_gt_duration = get_gt_token_duration(target_dur, valid_gt_trn) # ground truth start and end time for each word in utterance

        hyp_duration = []
        for token in valid_hyp_trn:
            cam_token = cams_dict[token[0]] #[1, :]
            time = np.linspace(0, cam_token.shape[0], num=cam_token.shape[0])
            fit = interp1d(time, cam_token, axis=0)
            new_time = np.linspace(0, cam_token.shape[0], num=800)
            cam_padded = fit(new_time)
            cam_unpadded = cam_padded[:input_length]

            if not np.any(cam_unpadded):
                continue
        
            peaks, _ = find_peaks(cam_unpadded, prominence=0.01, width=0.2)
            try:
                token_max_frame = np.max(peaks)
                hyp_duration.append((token_max_frame, token[0]))

            except:
                pass

        l_analysis = get_localisation_metric_count(hyp_duration, token_gt_duration)
        l_n_tp += l_analysis[0]
        l_n_fp += l_analysis[1]
        l_n_fn += l_analysis[2]
        
    # Compute precision, recall and fscore for detection task
    d_precision, d_recall, d_fscore = eval_detection_prf(d_n_tp, d_n_tp_fp, d_n_tp_fn)

    # Compute precision, recall and fscore for localisation task
    l_precision, l_recall, l_fscore = eval_localisation_prf(l_n_tp, l_n_fp, l_n_fn)


    # Print status
    print
    print("-"*79)
    print("Sigmoid threshold: {:.2f}".format(args.test_threshold))
    print("No. predictions:", d_n_tp_fp)
    print("No. true tokens:", d_n_tp_fn)
    print("Precision: {} / {} = {:.4f}%".format(d_n_tp, d_n_tp_fp, d_precision*100.))
    print("Recall: {} / {} = {:.4f}%".format(d_n_tp, d_n_tp_fn, d_recall*100.))
    print("F-score: {:.4f}%".format(d_fscore*100.))
    print("-"*79)

    print
    print("-"*79)
    print("LOCALISATION SCORES: ")
    print("Sigmoid threshold: {:.2f}".format(args.test_threshold))
    print("No. predictions:", l_n_fp)
    print("No. true tokens:", l_n_fn)
    print("Precision: {} / {} = {:.4f}%".format(l_n_tp, (l_n_tp + l_n_fp), l_precision*100.))
    print("Recall: {} / {} = {:.4f}%".format(l_n_tp, (l_n_tp + l_n_fn), l_recall*100.))
    print("F-score: {:.4f}%".format(l_fscore*100.))
    print("-"*79)


    


# python test_cnnpool.py --model_path 1620295553_cnnpool --target_type bow --test_threshold 0.4
    
