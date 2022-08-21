import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from dataset import ICLEVR_dataset
from model import Generator
from evaluator import evaluation_model
from utils import concat_test, concat_image


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
    parser.add_argument('--model_dir', default='./models/WGAN_epoch74_acc0.5694_1.pt', help='directory for the best model')
    parser.add_argument('--file_root', default='.', help='root directory for json file')
    parser.add_argument('--img_root', default='../../iclevr', help='root directory for png images')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--seed', type=int, default=999, help='manual seed')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--cuda', default=True, action='store_true')  # use cuda

    args = parser.parse_args()
    return args


def test(n_z, cond, netG, evaluator, device, img_name):
    'Test the model.'
    condition = cond.to(device, dtype=torch.float32)
    b_size = len(condition)
    best_acc = 0

    print("Sart Testing...")
    # For each epoch
    for i in range(5):
        ## Evaluating
        netG.eval()
        acc = 0
        z = torch.randn(b_size, n_z, 1, 1, device=device, dtype=torch.float32)
        # saving G's output on fixed_noise, generate image
        with torch.no_grad():
            gene_x = netG(z, condition).detach()
        # evaluate the generated image
        acc = evaluator.eval(gene_x, condition)
        # Save the best generator
        if acc > best_acc:
            best_x = gene_x
            best_acc = acc
    concat_image(best_x, save_name=f"{img_name}_acc{best_acc:.4f}")
    print(f"The best testing score = {best_acc:.4f}")        


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

    # Fix random seed
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load the training and testing dataset
    new_test_dataset = ICLEVR_dataset(file_path=args.file_root, img_path=args.img_root, mode="test")
    new_test_loader = DataLoader(new_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    new_test_cond = concat_test(new_test_loader)

    test_dataset = ICLEVR_dataset(file_path=args.file_root, img_path=args.img_root, mode="eval")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_cond = concat_test(test_loader)

    # Create the generator
    netG = Generator(args).to(device)
    # Load the best model
    netG.load_state_dict(torch.load(args.model_dir))
    # print(netG)

    # Create the evaluator
    evaluator = evaluation_model()

    # Test the model
    print("test.json: ")
    test(args.n_z, test_cond, netG, evaluator, device, "WGAN_test_results_1")
    print("new_test.json: ")
    test(args.n_z, new_test_cond, netG, evaluator, device, "WGAN_new_test_results_1")


if __name__ == "__main__":
    main()
