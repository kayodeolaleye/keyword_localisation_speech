from time import monotonic
import torch
import numpy as np
from tqdm import tqdm
import time
import calendar
from numba.cuda import args
from os import path
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from config import device, num_workers, print_freq, trained_model_dir
from data_gen import Flickr8kDataset, pad_collate
from models.cnnattend import CNNAttend
from models.optimizer import PSCOptimizer

from utils import ensure_folder, get_logger, parse_args, save_checkpoint, AverageMeter, write_hist_to_tb, write_scalar_to_tb

def train_net(args):
    torch.manual_seed(42)
    np.random.seed(42)

    model_id = str(calendar.timegm(time.gmtime())) + "_cnnattend_" + args.target_type
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
        model = CNNAttend(args.vocab_size, args.embed_size)
        print(model)
        model.to(device)

        # optimizer = PSCOptimizer(torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate, pin_memory=True, shuffle=True, num_workers=num_workers)
    valid_dataset = Flickr8kDataset('dev')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=pad_collate, pin_memory=True, shuffle=False, num_workers=num_workers)

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        train_loss = train(train_loader, model=model, optimizer=optimizer, epoch=epoch, logger=logger, target_type=args.target_type)
        
        # lr = optimizer.lr
        # print('\nLearning rate: {}'.format(lr))
        # step_num = optimizer.step_num
        # print('Step num: {}\n'.format(step_num))

        # One epoch's validation
        valid_loss, valid_precision, valid_recall, valid_fscore = valid(valid_loader=valid_loader, model=model, logger=logger, threshold=args.val_threshold)
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
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(
            epoch, epochs_since_improvement, model, optimizer, best_loss, is_best_loss, 
            best_precision, is_best_precision, best_recall, is_best_recall, best_fscore, 
            is_best_fscore, model_path)


    
    torch.save(model.state_dict(), path.join(model_path, "model.pth"))

def train(train_loader, model, optimizer, epoch, logger, target_type):
    model.train()
    losses = AverageMeter()

    # Create loss function
    criterion = nn.BCELoss()
    # Batches
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        target = None
        padded_input, bow_target, soft_target, _, input_lengths = data
        padded_input = padded_input.to(device)
        input_lengths = input_lengths.to(device)
        if target_type == 'bow':
            target = bow_target.to(device)
        elif target_type == 'soft':
            target = soft_target.to(device)
        else:
            print("Incorrect supervision's target. Choose either 'bow' or 'soft'.")
            break

        # Forward prop.
        out, attention_Weights = model(padded_input)
        loss = criterion(torch.sigmoid(out), target)

        # Back prop.
        optimizer.zero_grad()
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
    criterion = nn.BCELoss()

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
        loss = criterion(torch.sigmoid(out), target)

        # Keep track of metrics
        losses.update(loss.item())
        sigmoid_out = torch.sigmoid(out).cpu()
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


if __name__ == '__main__':
    main()


# python train_cnnattend.py --target_type bow --val_threshold 0.4 --vocab_size 1000 --embed_size 1000 --epochs 25
# tensorboard --logdir=runs