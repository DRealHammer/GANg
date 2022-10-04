import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from definitions import clamped_log, weights_init


import matplotlib.pyplot as plt

class Generator(nn.Module):
  
    def __init__(self, output_shape) -> None:
        super(Generator, self).__init__()

        self.input_res = 4
        self.input_length = 100

        # (channel, x, y)
        self.output_shape = output_shape
        self.output_channels = output_shape[0]

        # check for symmetrical image
        assert(self.output_shape[1] == self.output_shape[2])
        self.output_res = self.output_shape[1]
        
        self.initial = nn.Sequential(
            nn.Linear(self.input_length, self.input_res**2 *512),
            nn.BatchNorm1d(self.input_res**2 *512),
            nn.ReLU()
        )
      
        if self.output_res == 28:

            self.conv = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),

                nn.ConvTranspose2d(256, 128, 4, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),

                nn.ConvTranspose2d(128, 64, 4, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                nn.ConvTranspose2d(64, self.output_channels, 4, 2, bias=False),
                nn.Tanh()
            )

        elif self.output_res == 32:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),

                nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),

                nn.ConvTranspose2d(128, 64, 4, 2, padding=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                nn.ConvTranspose2d(64, self.output_channels, 4, 2, padding=3, bias=False),
                nn.Tanh()
            )

        elif self.output_res == 64:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),

                nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),

                nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                nn.ConvTranspose2d(64, self.output_channels, 4, 2, padding=1, bias=False),
                nn.Tanh()
            )


    def forward(self, x):
        pre_conv = self.initial(x).view(-1, 512, self.input_res, self.input_res)
        return self.conv(pre_conv)


class Discriminator(nn.Module):

    def __init__(self, input_shape, output_activation: nn.Module = nn.Sigmoid) -> None:
        super(Discriminator, self).__init__()
        slope = 0.2

        self.input_shape = input_shape

        self.input_channels = self.input_shape[0]

        # check for symmetrical image
        assert(self.input_shape[1] == self.input_shape[2])
        self.input_res = self.input_shape[1]
        
        if self.input_res == 28:
        
            self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 4, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(slope),

            nn.Conv2d(64, 128, 4, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(slope),

            nn.Conv2d(128, 256, 4, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(slope),

            nn.Conv2d(256, 512, 4, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(slope)
            )

        elif self.input_res == 32:
            self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 4, 2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(slope),

            nn.Conv2d(64, 128, 4, 2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(slope),

            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(slope),

            nn.Conv2d(256, 512, 4, 1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(slope)
            )

        elif self.input_res == 64:
            self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(slope),

            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(slope),

            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(slope),

            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(slope)
            )

        self.lin = nn.Sequential(
            nn.Linear(4*4*512, 1),
            output_activation()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.002, betas=(0.5, 0.999))

    def forward(self, x):
        x_2d = x.view(-1, self.input_channels, self.input_res, self.input_res)
        conv = self.conv(x_2d).view(-1, 4*4*512)
        return self.lin(conv)

    def loss(self, real, fake):
        dx = self.forward(real)
        dgz = self.forward(fake)
        l = - torch.mean(clamped_log(dx) + clamped_log(1-dgz))
        return l


    def fit(self, real, fake):
        self.zero_grad()
        loss = self.loss(real, fake)
        loss.backward()
        self.optimizer.step()
        return loss


class GAN(nn.Module):
    def __init__(self, img_shape: tuple) -> None:
        super(GAN, self).__init__()

        self.img_shape = img_shape
        self.generator = Generator(self.img_shape)
        self.generator.apply(weights_init)
        self.discriminator = Discriminator(self.img_shape)
        self.discriminator.apply(weights_init)

    def forward(self):
        pass

    def create_samples(self, n_samples: int, device='cpu'):
        z = torch.from_numpy(np.random.uniform(-1, 1, size=(n_samples, self.generator.input_length))).float().to(device)
        #z = torch.randn(size=(n_samples, self.generator.input_length)).float().to(device)
        #z = torch.from_numpy(np.random.normal(0.0, 1.0, size=(n_samples, self.generator.input_length))).float().to(device)
        self.generator.to(device)
        return self.generator.forward(z)

    def fit(self, data_loader: DataLoader, device):
            
        g_loss = 0
        d_loss = 0
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002) #betas=(0.5, 0.999))

        for batch_num, batch in enumerate(data_loader):

            fake_data = self.create_samples(len(batch[0]), device)

            # Train the discriminator
            real_data = batch[0].to(device)

            discriminator_loss = self.discriminator.fit(real_data, fake_data.detach())

            # Train the generator
            # TODO test if still good if old fake data is used
            fake_data = self.create_samples(len(batch[0]), device)
            dgz = self.discriminator.forward(fake_data)

            self.generator.zero_grad()
            gen_loss = self.generator_loss(dgz)
            gen_loss.backward()
            generator_optimizer.step()
            #self.generator.optimizer.step()
            
            g_loss += gen_loss
            d_loss += discriminator_loss

            if batch_num % 100 == 0:
                print(f'batch {batch_num} gen loss: {gen_loss}, discr loss: {discriminator_loss}')
                plt.imshow((fake_data[0].reshape(self.img_shape).detach().to('cpu').swapaxes(0, 1).swapaxes(1, 2) + 1) / 2)
        return g_loss, d_loss

    def generator_loss(self, dgz):
        return - torch.mean(clamped_log(dgz))