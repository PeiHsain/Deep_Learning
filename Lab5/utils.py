import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image


def concat_test(test_loader):
    'Concate all testing data conditions.\nOutput: all testing data conditions'
    first = True
    for cond in test_loader:
        if first == True:
            eval_cond = cond
            first = False
        else:
            eval_cond = torch.cat((eval_cond, cond))
    return eval_cond


def denormal_image(img):
    'Apply de-normalization while generating RGB images.'
    image = img.mul([0.5, 0.5, 0.5]).add_([0.5, 0.5, 0.5])
    return image


def concat_image(img, save_name=''):
    'Concate the generated images and save it.'
    img = denormal_image(img)
    grid = make_grid(img, nrow=8)
    save_image(grid, fp=os.path.join('./results', f"{save_name}.png"))


def plot_train_curve(g_loss, d_loss, acc, save_name=''):
    'Plot the Generator and Discriminator loss.'
    epoch = len(g_loss)
    fig, loss = plt.subplots()
    score = loss.twinx()
    plt.title(f"Loss curve", fontsize=15)
    loss.set_xlabel(f"Epochs")
    loss.set_ylabel(f"Loss")
    score.set_ylabel(f"Accuracy")
    curve1, = loss.plot(range(epoch), g_loss, color='lime', label='G loss')
    curve2, = loss.plot(range(epoch), d_loss, color='darkorange', label='D loss')
    curve3, = score.plot(range(epoch), acc, color='blue', label='Accuracy')
    curves = [curve1, curve2, curve3]
    loss.legend(curves, [curve.get_label() for curve in curves], loc='right')
    plt.savefig(os.path.join('./results', f"{save_name}.png"))

