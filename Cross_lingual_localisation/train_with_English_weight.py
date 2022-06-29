from distutils.command.config import config
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

import torch
import numpy as np
from tqdm import tqdm
import time
import calendar
import random
from os import path
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from config import device, num_workers, print_freq, trained_model_dir
from data_gen import Flickr8kDataset, pad_collate
from models.cnnattend import CNNAttend
from utils import ensure_folder, get_logger, parse_args, save_checkpoint, AverageMeter, write_hist_to_tb, write_scalar_to_tb

english_trained_model_dir = "/home/kayode/KAYODE/PhD/keyword_localisation_speech/Interspeech_no_Trimming/trained_models/1652979622_cnnattend_soft_100"

# wandb.init(project='Cross-lingual Keyword Localisation', entity="collarkay")
# sweep_config_fn = "config.yaml"
# sweep_id = wandb.sweep(sweep_config_fn, project="pytorch-sweeps-demo")
def train_net(args, config=None):
    
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
  
    model_id = str(calendar.timegm(time.gmtime())) + "_cnnattend_" + args.target_type # + "_random_data"
    print("Model ID: ", model_id)
    model_path = path.join(trained_model_dir, model_id)
    ensure_folder(model_path)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    best_precision = float('-inf')
    best_recall = float('-inf')
    best_fscore = float('-inf')
    writer = SummaryWriter(log_dir=path.join("runs", model_id))
    epochs_since_improvement = 0

    # Initialise / load checkpoint
    if checkpoint is None:
        # model
        model = CNNAttend(args.vocab_size, args.embed_size, args.fc_layer_size, args.dropout)
        print(model)
        
        # Load the English model
        model.load_state_dict(torch.load(path.join(english_trained_model_dir, "model.pth")))
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    
    logger = get_logger()

    # Custom dataloaders
    train_dataset = Flickr8kDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate, pin_memory=True, shuffle=True)
    valid_dataset = Flickr8kDataset('dev') # use all the available dev set
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=pad_collate, pin_memory=True, shuffle=False)

    # with wandb.init(config=config):
        # config = wandb.config
        # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        train_loss = train(train_loader, model=model, optimizer=optimizer, epoch=epoch, logger=logger, target_type=args.target_type)

        # One epoch's validation
        valid_loss, valid_precision, valid_recall, valid_fscore = valid(valid_loader=valid_loader, model=model, logger=logger, threshold=args.val_threshold)

        # wandb.log({"train_loss": train_loss, "valid_loss": valid_loss, "valid_precision": valid_precision, "valid_recall": valid_recall, "valid_fscore": valid_fscore})
        write_scalar_to_tb(
            writer,
            # lr,
            epoch,
            train_loss,
            valid_loss,
            valid_precision,
            valid_recall,
            valid_fscore)
        
        write_hist_to_tb(writer, model, epoch)

        # Check if there was an improvement
        is_best_loss = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)

        is_best_precision = valid_precision > best_precision
        best_precision = max(valid_precision, best_precision)

        is_best_recall = valid_recall > best_recall
        best_recall = max(valid_recall, best_recall)

        is_best_fscore = valid_fscore > best_fscore
        best_fscore = max(valid_fscore, best_fscore)

        if not(is_best_loss or is_best_precision or is_best_recall or is_best_fscore):
            epochs_since_improvement += 1
            # print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(
            epoch, epochs_since_improvement, model, optimizer, best_loss, is_best_loss, 
            best_precision, is_best_precision, best_recall, is_best_recall, best_fscore, 
            is_best_fscore, model_path)


    
    torch.save(model.state_dict(), path.join(model_path, "model.pth"))
    print("Model ID: ", model_id)

def train(train_loader, model, optimizer, epoch, logger, target_type):
    model.train()
    losses = AverageMeter()
    # word_freq = np.load("data/word_freq_train.npy")
    # weighted_word_freq = torch.from_numpy(np.array([i/np.sum(word_freq) for i in word_freq]))

    # Create loss function
    # weighted_word_freq = weighted_word_freq.to(device)
    # print("weight shape: ", weighted_word_freq.shape)
    # print("model.conv_module[0].weight", model.conv_module[0].weight[0][0][:2])
    criterion = nn.BCEWithLogitsLoss()
    # Batches
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        optimizer.zero_grad()
        target = None
        padded_input, bow_target, soft_target, _, input_lengths = data
        # print("padded input: ", padded_input[0][0][:2])
        padded_input = padded_input.to(device)
        input_lengths = input_lengths.to(device)
        if target_type == 'bow':
            target = bow_target.to(device)
            continue
        elif target_type == 'soft':
            target = soft_target.to(device)
        else:
            print("Incorrect supervision's target. Choose either 'bow' or 'soft'.")
            break

        # Forward prop.
        out, attention_Weights = model(padded_input)
        # loss = criterion(torch.sigmoid(out), target)
        # print("Out Shape: ", out.shape)
        # print("target shape: ", target.shape)
        loss = criterion(out, target)

        # Back prop.
        loss.backward(retain_graph=True)

        # update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

    return losses.avg


def valid(valid_loader, model, logger, threshold):
    model.eval()
    losses = AverageMeter()
    n_tp = 0  
    n_tp_fp = 0 # (tp + fp)
    n_tp_fn = 0 # (tp + fn)

    # Create loss function
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    # Batches
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input, bow_target, _, __, input_lengths = data
        # padded_input = torch.transpose(padded_input, 2, 1)
        padded_input = padded_input.to(device)
        input_lengths = input_lengths.to(device)
        target = bow_target.to(device)
        # Forward prop.
        out, attention_weights = model(padded_input)
        # loss = criterion(torch.sigmoid(out), target)
        loss = criterion(out, target)

        # Keep track of metrics
        losses.update(loss.item())
        sigmoid_out = torch.sigmoid(out).cpu()
        # sigmoid_out = out.cpu()
        sigmoid_out_thresholded = torch.ge(sigmoid_out, threshold).float()
        n_tp += torch.sum(sigmoid_out_thresholded * target.cpu()).numpy()
        n_tp_fp += torch.sum(sigmoid_out_thresholded).numpy()
        n_tp_fn += torch.sum(target.cpu()).numpy()

    precision = n_tp / n_tp_fp
    recall = n_tp / n_tp_fn
    fscore = 2 * precision * recall / (precision + recall)

    # Print status
    logger.info('\nValidation Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(loss=losses))
    logger.info('\nValidation Precision: {precision:.4f}\n'.format(precision=precision))
    logger.info('\nValidation Recall: {recall:.4f}\n'.format(recall=recall))
    logger.info('\nValidation F-score: {fscore:.4f}\n'.format(fscore=fscore))
    
    return losses.avg, precision, recall, fscore



def main():
    global args    
    args = parse_args()
    train_net(args)

    # Run training script max_iter times sequentially
    # iter = 0
    # max_iter = 20
    # while iter <= max_iter:
    #     train_net(args)
    #     iter += 1

if __name__ == '__main__':
    main()
