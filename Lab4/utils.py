import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image


def kl_criterion(mu, logvar, args):
    'KL divergence loss.'
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= args.batch_size  
    return KLD


def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr


def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err


# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr


def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def pred(x, cond, modules, args):
    'Predict the results.'
    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    pred_seq = []

    # x -> 30 frame(batch, frame, 64x64)
    # time 0
    x_past = x[:, 0]
    pred_seq.append(x_past.detach().cpu().numpy())

    for i in range(1, args.n_eval):
        # input time t-1 to predict, x -> encoder
        c_t = cond[:, i, :]
        h_past, skip_log = modules['encoder'](x_past)

        # skip connection neural advection model
        if args.last_frame_skip or i < args.n_past:
            skip = skip_log

        if i < args.n_past: # get real test image to connect
            # time t
            h_t, _ = modules['encoder'](x[:, i])
            z_t, mu, logvar = modules['posterior'](h_t)
            # time t-1, predict
            z_cond_cat = torch.cat((h_past, z_t, c_t), axis=1)
            f_pred = modules['frame_predictor'](z_cond_cat)
            x_past = x[:, i]
        else: # predict last frame, time t-1
            # get z by normal distribution
            z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()
            # get latent vector
            z_cond_cat = torch.cat((h_past, z_t, c_t), axis=1)
            f_pred = modules['frame_predictor'](z_cond_cat)
            # latent vector -> decoder
            x_past = modules['decoder']((f_pred, skip))

        pred_seq.append(x_past.detach().cpu().numpy())
    # (n_eval, batch_size, 3, 64, 64) -> (batch_size, n_eval, 3, 64, 64)
    # print(len(pred_seq))
    pred_seq = torch.tensor(np.array(pred_seq)).permute(1, 0, 2, 3, 4)
    return pred_seq


def plot_pred(x, cond, modules, epoch, args):
    'Plot the predicted images.'
    prediction = pred(x, cond, modules, args)
    os.makedirs(f'{args.log_dir}/epoch{epoch}', exist_ok=True)
    save_pred(prediction[0], epoch, args.log_dir)
    save_gt_gif(x[0], epoch, args.log_dir, args.n_eval)
    gif_cat(epoch, args.log_dir)


def save_pred(x, epoch, path):
    'Save the predicted image.'
    # x size (frame, 3, 64, 64) -> (frame, 64, 64, 3)
    x = x.permute(0, 2, 3, 1).numpy()
    # print("pred img: ", x[0])
    for i in range(len(x)):
        x[i] *= 255
        img = Image.fromarray(np.uint8(x[i]))
        img.save(os.path.join(path, f'epoch{epoch}/{i}.png'))
    save_gif(epoch, path)


def save_gif(epoch, path):
    'Save the gif image of the prediction.'
    image_list = []
    # the path of saved image
    file_name = os.path.join(path, f'epoch{epoch}')
    # take all .png imag
    for image_name in os.listdir(file_name):
        if image_name.endswith('.png'):
            image_list.append(image_name)
    # print(image_list)
    # print(int(image_list[0].split('.')[0]))
    # convert the string before "." into the number and be as a key to sort
    image_list.sort(key=lambda x: int(x.split('.')[0]))

    # the name of .gif
    gif_name = os.path.join(file_name, f'pred.gif')
    # create gif
    frames = []
    for im in image_list:
        if im.endswith('.png'):
            im = os.path.join(file_name, im)
            frames.append(imageio.imread(im))
        else:
            pssrint("This image not .png => "+ im)

    # duration, set image changing time (sec)
    imageio.mimsave(gif_name,frames,'GIF',duration = 0.3)


def save_gt_gif(x, epoch, path, frame):
    'Save the gif image of the ground truth.'
    x = x.permute(0, 2, 3, 1).cpu().numpy()
    print("gt img: ", x[0])
    # the path of saved image
    file_name = os.path.join(path, f'epoch{epoch}')
    # the name of .gif
    gif_name = os.path.join(file_name, f'gt.gif')
    # create gif
    frames = []
    for i in range(frame):
        x[i] *= 255
        img = Image.fromarray(np.uint8(x[i]))
        frames.append(img)
    # duration, set image changing time (sec)
    imageio.mimsave(gif_name,frames,'GIF',duration = 0.3)


def gif_cat(epoch, path):
    'Concatenate the gif image of the prediction and the ground truth.'
    file_name = os.path.join(path, f'epoch{epoch}')
    # opening up of images
    img1 = Image.open(os.path.join(file_name, f'gt.gif'))
    img2 = Image.open(os.path.join(file_name, f'pred.gif'))

    # creating a new image and pasting 
    cat_img = Image.new("RGB", (img1.width + img2.width, img1.height))
    # pasting the first image (image_name, (position))
    cat_img.paste(img1, (0, 0))
    # pasting the second image (image_name, (position))
    cat_img.paste(img2, (img1.width, 0))
    # save image
    cat_img.save(os.path.join(file_name, f'concat.gif'))



def plot_KL(kld, beta, dir):
    'Plot the KL loss and beta curve.'
    epoch = len(kld)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title(f"KL curve", fontsize=15)
    ax1.set_xlabel(f"Epochs")
    ax1.set_ylabel(f"Loss")
    ax2.set_ylabel(f"Beta (0~1)")
    curve1, = ax1.plot(range(epoch), kld, color='lime', label='KL loss')
    curve2, = ax2.plot(range(epoch), beta, color='darkorange', label='KL beta')
    curves = [curve1, curve2]
    ax1.legend(curves, [curve.get_label() for curve in curves], loc='upper right')
    plt.savefig(os.path.join(dir, f'kl_1.jpg'))


def plot_PSNR(psnr, dir):
    'Plot the PSNR curve.'
    fig = plt.figure()
    plt.title(f"PSNR curve", fontsize=15)
    plt.xlabel(f"Epochs (every 5 epochs)")
    plt.ylabel(f"PSNR")
    plt.plot(range(len(psnr)), marker='o')
    plt.savefig(os.path.join(dir, f'psnr_1.jpg'))


def plot_figure(KLD, KLB, TFR, MSE, LOSS, args):
    'Plot the figure include all results.'
    epoch = args.niter
    fig, ratio = plt.subplots()
    loss = ratio.twinx()
    plt.title(f"Compare figure", fontsize=15)
    ratio.set_xlabel(f"Epochs")
    ratio.set_ylabel(f"Ratio (0~1)")
    loss.set_ylabel(f"Loss")

    curve1, = ratio.plot(range(epoch), KLB, color='darkorange', linestyle='dashed', label='KL beta')
    curve2, = ratio.plot(range(epoch), TFR, color='lime', linestyle='dashed', label='Teacher force ratio')
    curve3, = loss.plot(range(epoch), KLD, color='blue', label='KL loss')
    curve4, = loss.plot(range(epoch), MSE, color='brown', label='MSE loss')
    curve5, = loss.plot(range(epoch), LOSS, color='black', label='Total loss')

    curves = [curve1, curve2, curve3, curve4, curve5]
    ratio.legend(curves, [curve.get_label() for curve in curves], loc='upper right')
    plt.savefig(os.path.join(args.log_dir, f'figure_1.jpg'))
