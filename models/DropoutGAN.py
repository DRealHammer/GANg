import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from definitions import clamped_log, weights_init
from models.DCGAN import Generator, Discriminator as CDiscriminator

import matplotlib.pyplot as plt

class Discriminator(nn.Module):

    def __init__(self, n_discriminators, img_shape):
        super(Discriminator, self).__init__()

        self.discriminators = nn.ModuleList([CDiscriminator(img_shape) for _ in range(n_discriminators)])
        self.n_discriminators = n_discriminators
        self.optimizer = torch.optim.Adam(self.discriminators.parameters(), lr=0.002, betas=(0.5, 0.999))
        
    def forward(self, x):
        res = torch.zeros(self.n_discriminators, len(x), 1)
        for i, discr in enumerate(self.discriminators):
            res[i] = discr.forward(x)
        return res
    
    def loss(self, real, fake):
        assert(len(real) == len(fake))
        res = torch.zeros(self.n_discriminators)
        
        for i, discr in enumerate(self.discriminators):
            res[i] = discr.loss(real, fake)
        return res
        
    def fit(self, real, fake):
        self.zero_grad()
        losses = self.loss(real, fake)
        l = torch.sum(losses)
        l.backward()
        self.optimizer.step()
        return l


class GAN(nn.Module):
    def __init__(self, img_shape: tuple, n_discriminators = 5, p_drop = 0.5) -> None:
        super(GAN, self).__init__()


        self.img_shape = img_shape
        self.generator = Generator(self.img_shape)
        self.generator.apply(weights_init)
        self.discriminator = Discriminator(n_discriminators, self.img_shape)
        self.discriminator.apply(weights_init)
        self.p_drop = p_drop

    def forward(self):
        pass

    def create_samples(self, n_samples: int, device='cpu'):
        z = torch.from_numpy(np.random.uniform(-1, 1, size=(n_samples, self.generator.input_length))).float().to(device)
        #z = torch.randn(size=(n_samples, self.generator.input_length)).float().to(device)
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
            dgz = self.discriminator.forward(fake_data)

            # drop some results
            ones = torch.ones(len(dgz))
            active = torch.binomial(ones, ones * self.p_drop).bool()

            # if no discriminator is active pick one randomly to have a gradient
            # else use the active ones
            if torch.count_nonzero(active) == 0:
                dgz = dgz[torch.randint(high=len(dgz), size=(1,)).item()]
            else:
                dgz = dgz[active]

            self.generator.zero_grad()
            gen_loss = self.generator_loss(dgz)
            gen_loss.backward()
            generator_optimizer.step()
            
            g_loss += gen_loss
            d_loss += discriminator_loss

            if batch_num % 100 == 0:
                print(f'batch {batch_num} gen loss: {gen_loss}, discr loss: {discriminator_loss}')
                plt.imshow((fake_data[0].reshape(self.img_shape).detach().to('cpu').swapaxes(0, 1).swapaxes(1, 2) + 1) / 2)
                plt.show()
        return g_loss, d_loss

    def generator_loss(self, dgz):
        losses = - torch.mean(clamped_log(dgz), dim=1)
        return torch.sum(losses)