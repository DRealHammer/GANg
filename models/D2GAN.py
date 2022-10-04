import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from definitions import clamped_log, weights_init
from models.DCGAN import Generator, Discriminator as CDiscriminator

import matplotlib.pyplot as plt

class Discriminator(nn.Module):

    def __init__(self, input_shape, alpha=0.2, beta=0.1):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        self.alpha = alpha
        self.beta = beta
        self.d1 = CDiscriminator(self.input_shape, nn.Softplus)
        self.d2 = CDiscriminator(self.input_shape, nn.Softplus)

        self.optimizer1 = torch.optim.Adam(self.d1.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer2 = torch.optim.Adam(self.d2.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def forward(self, x):
        return self.d1.forward(x), self.d2.forward(x)

  
    def loss(self, real, fake):
        d1x, d2x = self.forward(real)
        d1gz, d2gz = self.forward(fake)
        
        d1_loss = torch.mean(self.alpha * clamped_log(d1x) - d1gz)
        d2_loss = torch.mean(-d2x + self.beta * clamped_log(d2gz))

        return - d1_loss, - d2_loss

    def fit(self, real, fake):
        self.d1.zero_grad()
        self.d2.zero_grad()

        loss1, loss2 = self.loss(real, fake)

        loss1.backward()
        loss2.backward()

        self.optimizer1.step()
        self.optimizer2.step()

        return loss1 + loss2


class GAN(nn.Module):
    def __init__(self, img_shape: tuple, alpha=0.2, beta=0.1) -> None:
        super(GAN, self).__init__()

        self.img_shape = img_shape
        self.generator = Generator(self.img_shape)
        self.generator.apply(weights_init)
        self.discriminator = Discriminator(self.img_shape, alpha, beta)
        self.discriminator.apply(weights_init)
        self.alpha = alpha
        self.beta = beta

    def forward(self):
        pass

    def create_samples(self, n_samples: int, device='cpu'):
        z = torch.from_numpy(np.random.uniform(-1, 1, size=(n_samples, self.generator.input_length))).float().to(device)
        #z = torch.randn(size=(n_samples, self.generator.input_length)).float().to(device)
        #z = torch.normal(mean=torch.zeros_like(z), std=torch.ones_like(z)*0.01).to(device)
        self.generator.to(device)
        return self.generator.forward(z)

    def fit(self, data_loader: DataLoader, device):
            

        g_loss = 0
        d_loss = 0
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))


        for batch_num, batch in enumerate(data_loader):

            fake_data = self.create_samples(len(batch[0]), device)

            # Train the discriminator
            real_data = batch[0].to(device)

            discriminator_loss = self.discriminator.fit(real_data, fake_data.detach())

            # Train the generator
            # TODO test if still good if old fake data is used
            fake_data = self.create_samples(len(batch[0]), device)
            d1gz, d2gz = self.discriminator.forward(fake_data)

            self.generator.zero_grad()
            gen_loss = self.generator_loss(d1gz, d2gz)
            gen_loss.backward()
            generator_optimizer.step()
            
            g_loss += gen_loss
            d_loss += discriminator_loss

            if batch_num % 100 == 0:
                print(f'batch {batch_num} gen loss: {gen_loss}, discr loss: {discriminator_loss}')
                plt.imshow((fake_data[0].reshape(self.img_shape).detach().to('cpu').swapaxes(0, 1).swapaxes(1, 2) + 1) / 2)
                plt.show()

        return g_loss, d_loss

    def generator_loss(self, d1gz, d2gz):
        d1_loss = torch.mean(-d1gz)
        d2_loss = torch.mean(clamped_log(d2gz))
        return d1_loss + self.beta * d2_loss