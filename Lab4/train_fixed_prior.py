import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred, finn_eval_seq, pred, plot_figure, plot_PSNR, plot_KL
from prefetch_generator import BackgroundGenerator

torch.backends.cudnn.benchmark = True


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    # parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--model_dir', default='./logs/fp/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000-kl-mode=True-4', help='base directory to save logs')
    parser.add_argument('--data_root', default='.', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=100, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=(1/200), help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0.1, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=True, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.1, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=10, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=12, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--cond_dim', type=int, default=7, help='dimensionality of condition')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  # use cuda

    args = parser.parse_args()
    return args


def train(x, cond, modules, optimizer, kl_anneal, args, epoch):
    # clear the gradient
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0
    use_teacher_forcing = True if random.random() < args.tfr else False
    # inference model
    h = [modules['encoder'](x[:, i]) for i in range(args.n_past + args.n_future)]
    
    for i in range(1, args.n_past + args.n_future):
        # at time t, x -> encoder
        x_gt = x[:, i]
        c_t = cond[:, i, :]
        h_t = h[i][0]

        # at time t-1, x -> encoder
        if not use_teacher_forcing and i > 1: # past prediction
            h_past, _ = modules['encoder'](x_pred)
        else: # ground truth
            h_past, _ = h[i-1]

        # skip connection neural advection model
        if args.last_frame_skip or i < args.n_past:
            _, skip = h[i-1]

        # reparameter
        z_t, mu, logvar = modules['posterior'](h_t)
        # predict frame, time t-1 for encoder
        # concatenate the condition part with the latent vector (CVAE), cond + Ht-1 + Zt
        z_cond_cat = torch.cat((h_past, z_t, c_t), axis=1)
        f_pred = modules['frame_predictor'](z_cond_cat)

        # latent vector -> decoder
        x_pred = modules['decoder']((f_pred, skip))
        # kl divergence loss
        kld += kl_criterion(mu, logvar, args)
        # MSE loss (reconstruction loss)
        mse += nn.MSELoss()(x_pred, x_gt)
        # raise NotImplementedError

    # kl cost annealing
    beta = kl_anneal.get_beta(epoch)
    # loss function
    loss = mse + kld * beta
    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past), beta


class kl_annealing():
    def __init__(self, args):
        super().__init__()
        # Annealing cycle time
        self.cycTime = args.niter / args.kl_anneal_cycle
        # Two mode -> monotonic, cyclical
        if args.kl_anneal_cyclical:
            self.mode = 'cyclical'
        else:
            self.mode = 'monotonic'
        # raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
    
    def get_beta(self, epoch):
        if self.mode == 'cyclical':
            # half cycle to keep 1
            cycEpoch = epoch % (self.cycTime)
            beta = (1. / self.cycTime) * cycEpoch * 2
            return beta if cycEpoch < (self.cycTime/2) else 1.
        else:
            # one cycle time to reach 1, later keep 1 to the end
            beta = (1. / self.cycTime) * epoch
            return beta if epoch < self.cycTime else 1.
        # raise NotImplementedError


def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    print("device: ", device)
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f-kl-mode=%s-4'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta, args.kl_anneal_cyclical)

        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim+args.cond_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- training loop ------------------------------------

    # os.makedirs("./results", exist_ok=True)
    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    # save the results of each epoch
    PSNR = [] # every 5 epoch
    KLD = []
    KLBeta = []
    TFR = []
    MSE = []
    Loss = []
    for epoch in range(start_epoch, start_epoch + niter):
        # training mode
        frame_predictor.train()
        posterior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0

        for _ in range(args.epoch_size):
            try:
                seq, cond = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)
            
            seq, cond = seq.to(device, dtype=torch.float32), cond.to(device, dtype=torch.float32)
            loss, mse, kld, beta = train(seq, cond, modules, optimizer, kl_anneal, args, epoch)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld
        
        if epoch >= args.tfr_start_decay_epoch:
            ### Update teacher forcing ratio ###
            if args.tfr >= (args.tfr_lower_bound+args.tfr_decay_step):
                # ratio decay by the step size
                args.tfr -= args.tfr_decay_step
            else:
                # reach the lower bound value
                args.tfr = args.tfr_lower_bound
            # raise NotImplementedError

        progress.update(1)
        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n' % (epoch, epoch_loss  / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size)))
        
        # save the results
        KLD.append(epoch_kld / args.epoch_size)
        KLBeta.append(beta)
        TFR.append(args.tfr)
        MSE.append(epoch_mse / args.epoch_size)
        Loss.append(epoch_loss  / args.epoch_size)

        # evaluate mode
        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

        if epoch % 5 == 0:
            psnr_list = []
            for _ in range(len(validate_data) // args.batch_size):
                try:
                    validate_seq, validate_cond = next(validate_iterator)
                except StopIteration:
                    validate_iterator = iter(validate_loader)
                    validate_seq, validate_cond = next(validate_iterator)

                validate_seq, validate_cond = validate_seq.to(device, dtype=torch.float32), validate_cond.to(device, dtype=torch.float32)
                pred_seq = pred(validate_seq, validate_cond, modules, args)
                _, _, psnr = finn_eval_seq(validate_seq[:, args.n_past:], pred_seq[:, args.n_past:])
                psnr_list.append(psnr)
                
            ave_psnr = np.mean(np.concatenate(psnr_list))

            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))
            # save the psnr
            PSNR.append(ave_psnr)

            if ave_psnr > best_val_psnr:
                best_val_psnr = ave_psnr
                # save the model
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'args': args,
                    'last_epoch': epoch},
                    '%s/model.pth' % args.log_dir)

        if epoch % 20 == 0:
            try:
                validate_seq, validate_cond = next(validate_iterator)
            except StopIteration:
                validate_iterator = iter(validate_loader)
                validate_seq, validate_cond = next(validate_iterator)
            validate_seq, validate_cond = validate_seq.to(device, dtype=torch.float32), validate_cond.to(device, dtype=torch.float32)
            plot_pred(validate_seq, validate_cond, modules, epoch, args)
    plot_figure(KLD, KLBeta, TFR, MSE, Loss, args)
    plot_PSNR(PSNR, args.log_dir)
    plot_KL(KLD, KLBeta, args.log_dir)


def demo():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    print("device: ", device)
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model.pth' % args.model_dir)
    optimizer = args.optimizer
    model_dir = args.model_dir
    data_root = args.data_root
    # niter = args.niter
    args = saved_model['args']
    args.optimizer = optimizer
    args.model_dir = model_dir
    # args.log_dir = '%s/continued' % args.log_dir
    args.data_root = data_root
    # start_epoch = saved_model['last_epoch']

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(args)

    # ------------ build the models  --------------
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']

    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)
    test_iterator = iter(test_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- testing loop ------------------------------------
    # evaluate mode
    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()

    loop_size = len(test_data) // args.batch_size
    psnr_list = []
    progress = tqdm(total=loop_size)
    for _ in range(loop_size):
        try:
            test_seq, test_cond = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_loader)
            test_seq, test_cond = next(test_iterator)

        test_seq, test_cond = test_seq.to(device, dtype=torch.float32), test_cond.to(device, dtype=torch.float32)
        pred_seq = pred(test_seq, test_cond, modules, args)
        _, _, psnr = finn_eval_seq(test_seq[:, args.n_past:], pred_seq[:, args.n_past:])
        psnr_list.append(psnr)
        progress.update(1)
        
    ave_psnr = np.mean(np.concatenate(psnr_list))
    print(f"Average score on testing dataset = {ave_psnr}")


if __name__ == '__main__':
    # main()
    demo()
