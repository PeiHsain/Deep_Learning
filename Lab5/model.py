import torch
import torch.nn as nn


def weights_init(m):
    'All model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02'
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


## DCGAN
class Generator(nn.Module):
    'Generator. Map the latent space vector (z) to data-space.'
    def __init__(self, args):
        super(Generator, self).__init__()
        self.z_dim = args.n_z
        self.c_dim = args.out_cond
        # condition embedding. 24 -> 100
        self.cond_embed = nn.Sequential(
            nn.Linear(args.n_cond, self.c_dim)
        )
        # input is Z and conditions (z_dim+c_dim)x1x1, going into a convolution
        self.conv1 = self.conv(self.z_dim + self.c_dim, args.n_Gf * 8, 4, 1, 0, False, True)
        # state size. (ngf*8) x 4 x 4
        self.conv2 = self.conv(args.n_Gf * 8, args.n_Gf * 4, 4, 2, 1, False, True)
        # state size. (ngf*4) x 8 x 8
        self.conv3 = self.conv(args.n_Gf * 4, args.n_Gf * 2, 4, 2, 1, False, True)
        # state size. (ngf*2) x 16 x 16
        self.conv4 = self.conv(args.n_Gf * 2, args.n_Gf, 4, 2, 1, False, True)
        # state size. (ngf) x 32 x 32
        self.output = nn.Sequential(
            nn.ConvTranspose2d(args.n_Gf, args.n_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def conv(self, in_channels, out_channels, kernel_size, stride, padding, bias, act_inplace):
        seq_modules = nn.Sequential(
            # convolution layer -> normalize
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=act_inplace),
        )
        return seq_modules

    def forward(self, input, condition):
        # input: z (batch, z_dim, 1, 1)
        z = input.view(-1, self.z_dim, 1, 1)
        # condition: one-hot (batch, 24)
        c = self.cond_embed(condition).view(-1, self.c_dim, 1, 1)
        # concate the z and c (-1, z_dim+c_dim, 1, 1)
        z_c_cat = torch.cat((z, c), dim=1) 
        gz = self.conv1(z_c_cat)
        gz = self.conv2(gz)
        gz = self.conv3(gz)
        gz = self.conv4(gz)
        gz = self.output(gz)
        return gz # output: G(z+c) (batch, 3, 64, 64)


class Discriminator(nn.Module):
    'Discriminator. A binary classification network that takes an image as input and outputs a scalar probability that the input image is real (as opposed to fake).'
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.imag_size = args.image_size
        # self.c_dim = args.out_cond
        # condition embedding. 24 -> 64x64
        self.cond_embed = nn.Sequential(
            nn.Linear(args.n_cond, self.imag_size*self.imag_size)
        )
        self.conv1 = nn.Sequential(
            # input is (nc+1) x 64 x 64
            nn.Conv2d(args.n_channel + 1, args.n_Df, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ndf) x 32 x 32
        self.conv2 = self.conv(args.n_Df, args.n_Df * 2, 4, 2, 1, False, 0.2, True)
        # state size. (ndf*2) x 16 x 16
        self.conv3 = self.conv(args.n_Df * 2, args.n_Df * 4, 4, 2, 1, False, 0.2, True)
        # state size. (ndf*4) x 8 x 8
        self.conv4 = self.conv(args.n_Df * 4, args.n_Df * 8, 4, 2, 1, False, 0.2, True)
        self.output = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(args.n_Df * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def conv(self, in_channels, out_channels, kernel_size, stride, padding, bias, slop, act_inplace):
        seq_modules = nn.Sequential(
            # convolution layer -> normalize
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=slop , inplace=act_inplace),
        )
        return seq_modules        

    def forward(self, input, condition):
        # input: x (batch, 3, 64, 64) -> real image or G(z+c)
        x = input.view(-1, 3, self.imag_size, self.imag_size)
        # condition: one-hot (batch, 24)
        c = self.cond_embed(condition).view(-1, 1, self.imag_size, self.imag_size)
        # concate the x and c (-1, 3+1, 64, 64)
        x_c_cat = torch.cat((x, c), dim=1) 
        dx = self.conv1(x_c_cat)
        dx = self.conv2(dx)
        dx = self.conv3(dx)
        dx = self.conv4(dx)
        dx = self.output(dx)
        return dx.view(-1) # output: D(x+c) (batch)


## WGAN
class W_Discriminator(nn.Module):
    'Discriminator for WGAN. A binary classification network that takes an image as input and outputs a scalar probability that the input image is real (as opposed to fake).'
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.imag_size = args.image_size
        # self.c_dim = args.out_cond
        # condition embedding. 24 -> 64x64
        self.cond_embed = nn.Sequential(
            nn.Linear(args.n_cond, self.imag_size*self.imag_size)
        )
        self.conv1 = nn.Sequential(
            # input is (nc+1) x 64 x 64
            nn.Conv2d(args.n_channel + 1, args.n_Df, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ndf) x 32 x 32
        self.conv2 = self.conv(args.n_Df, args.n_Df * 2, 4, 2, 1, False, 0.2, True)
        # state size. (ndf*2) x 16 x 16
        self.conv3 = self.conv(args.n_Df * 2, args.n_Df * 4, 4, 2, 1, False, 0.2, True)
        # state size. (ndf*4) x 8 x 8
        self.conv4 = self.conv(args.n_Df * 4, args.n_Df * 8, 4, 2, 1, False, 0.2, True)
        self.output = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(args.n_Df * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def conv(self, in_channels, out_channels, kernel_size, stride, padding, bias, slop, act_inplace):
        seq_modules = nn.Sequential(
            # convolution layer -> normalize
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=slop , inplace=act_inplace),
        )
        return seq_modules        

    def forward(self, input, condition):
        # input: x (batch, 3, 64, 64) -> real image or G(z+c)
        x = input.view(-1, 3, self.imag_size, self.imag_size)
        # condition: one-hot (batch, 24)
        c = self.cond_embed(condition).view(-1, 1, self.imag_size, self.imag_size)
        # concate the x and c (-1, 3+1, 64, 64)
        x_c_cat = torch.cat((x, c), dim=1) 
        dx = self.conv1(x_c_cat)
        dx = self.conv2(dx)
        dx = self.conv3(dx)
        dx = self.conv4(dx)
        dx = self.output(dx)
        return dx.view(-1) # output: D(x+c) (batch)