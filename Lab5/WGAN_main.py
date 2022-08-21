import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import copy
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ICLEVR_dataset
from model import weights_init, Generator, W_Discriminator
from evaluator import evaluation_model
from utils import concat_test, concat_image, plot_train_curve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.5, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--image_size', default=64, type=int, help='the spatial size of the images used for training.')
    parser.add_argument('--n_channel', default=3, type=int, help='number of channels in the training images')
    parser.add_argument('--n_z', default=100, type=int, help='size of z latent vector')
    parser.add_argument('--n_Gf', default=64, type=int, help='size of feature maps in generator')
    parser.add_argument('--n_Df', default=64, type=int, help='size of feature maps in discriminator')
    parser.add_argument('--n_cond', default=24, type=int, help='dimension of the one-hot conditions')
    parser.add_argument('--out_cond', default=200, type=int, help='dimension of the conditions embedding')
    parser.add_argument('--file_root', default='.', help='root directory for json file')
    parser.add_argument('--img_root', default='../../iclevr', help='root directory for png images')
    parser.add_argument('--seed', type=int, default=999, help='manual seed')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--cuda', default=True, action='store_true')  # use cuda

    args = parser.parse_args()
    return args


def train(args, train_loader, test_loader, netD, netG, evaluator, device):
    'Train the model.'

    # Set loss funtions and optimizers
    # criterion = nn.BCELoss()
    optimizerD = optim.RMSprop(netD.parameters(), lr=args.lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=args.lr)
    

    # All evaluating data conditions
    eval_cond = concat_test(test_loader)
    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(len(eval_cond), args.n_z, 1, 1, device=device)

    # G_loss = []
    # D_loss = []
    eval_acc = []
    best_acc = 0

    print("Sart Training...")
    # For each epoch
    for epoch in range(args.epoch_size):
        # g_loss_total = 0
        # d_loss_total = 0
        ## Training
        netD.train()
        netG.train()
        # For each batch in the dataloader
        for img, cond in tqdm(train_loader):
            real_img = img.to(device, dtype=torch.float32)
            condition = cond.to(device, dtype=torch.float32)
            b_size = len(img)
            # no log in loss
            one = torch.cuda.FloatTensor(np.ones(b_size))
            mone = -1 * one

            # make sure Lipschitz continuity
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)
            """
            1. Update D -> maximze log(D(x)) + log(1 - D(G(z)))
            """
            ## Train with real batch
            optimizerD.zero_grad()
            D_output = netD(real_img, condition) # forward pass
            # errD_real = criterion(D_output, r_label) # calculate loss of D on real batch
            D_output.backward(one) # calculate gradient of D in backward pass

            ## Train with fake batch
            noise = torch.randn(b_size, args.n_z, 1, 1, device=device, dtype=torch.float32)
            fake_img = netG(noise, condition) # generate fake image batch
            D_output = netD(fake_img.detach(), condition) # classify fake batch
            # errD_fake = criterion(D_output, f_label) # calculate loss of D on fake batch
            D_output.backward(mone) # calculate the gradient for this batch, sum with pre-gradient
            # errD = errD_real + errD_fake # compute error of D as sum over the fake and real batches
            
            ## Update D
            optimizerD.step()

            """
            2. Updata G -> maximize log(D(G(z))) = minimize log(1 - D(G(z)))
            """
            for i in range(5):
                optimizerG.zero_grad()
                noise = torch.randn(b_size, args.n_z, 1, 1, device=device, dtype=torch.float32)
                fake_img = netG(noise, condition) # generate fake image batch
                D_output = netD(fake_img, condition) # perform another forward pass of fake image
                # errG = criterion(D_output, r_label) # calculate loss of G
                D_output.backward(one) # calculate gradients for G
                optimizerG.step() # Update G

            # d_loss_total += errD.item()
            # g_loss_total += errG.item()
        # print(f"G loss = {g_loss_total:.4f}, D loss = {d_loss_total:.4f}")
        # G_loss.append(g_loss_total)
        # D_loss.append(d_loss_total)

        ## Evaluating
        netD.eval()
        netG.eval()
        acc = 0
        condition = eval_cond.to(device, dtype=torch.float32)
        # saving G's output on fixed_noise, generate image
        with torch.no_grad():
            gene_x = netG(fixed_noise, condition).detach()
        # evaluate the generated image
        acc = evaluator.eval(gene_x, condition)
        eval_acc.append(acc)
        print(f"Epoch {epoch}: Evaluated score = {acc}")

        # Save the best generator
        if acc > best_acc:
            best_G = copy.deepcopy(netG.state_dict())
            best_acc = acc
            torch.save(best_G, os.path.join('./models', f'WGAN_epoch{epoch}_acc{best_acc}_1.pt'))
            concat_image(gene_x, save_name=f"WGAN_epoch{epoch}_acc{best_acc}_1")
    # plot_train_curve(G_loss, D_loss, eval_acc, save_name=f"WGAN_train_process_1")


def main():
    # Hyperparameter
    args = parse_args()

    # Check the GPU is avialible, else use the CPU
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    print("device: ", device)

    os.makedirs('%s/models' % args.file_root, exist_ok=True)
    os.makedirs('%s/results' % args.file_root, exist_ok=True)

    # Fix random seed
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load the training and testing dataset
    train_dataset = ICLEVR_dataset(file_path=args.file_root, img_path=args.img_root, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataset = ICLEVR_dataset(file_path=args.file_root, img_path=args.img_root, mode="eval")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create the generator
    netG = Generator(args).to(device)
    # Randomly initialize all weights
    netG.apply(weights_init)
    print(netG)

    # Create the discriminator
    netD = W_Discriminator(args).to(device)
    # Randomly initialize all weights
    netD.apply(weights_init)
    print(netD)

    # Create the evaluator
    evaluator = evaluation_model()

    # Train the model
    train(args, train_loader, test_loader, netD, netG, evaluator, device)


if __name__ == "__main__":
    main()
