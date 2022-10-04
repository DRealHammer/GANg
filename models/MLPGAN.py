import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from definitions import clamped_log

import matplotlib.pyplot as plt

class Generator(nn.Module):
  
    def __init__(self, input_length: int, output_shape: tuple, layer_sizes: list, activation_layer: nn.Module = nn.ReLU, activation_output: nn.Module = nn.Sigmoid) -> None:
        super(Generator, self).__init__()
        self.layer_sizes = layer_sizes
        self.model = nn.Sequential()
        self.input_length = input_length

        # (channel, x, y)
        self.output_length = np.prod(output_shape)
        self.output_shape = output_shape


        if len(layer_sizes) == 0:
            self.model.append(nn.Linear(input_length, self.output_length))
        
        else:
            self.model.append(nn.Linear(input_length, layer_sizes[0]))
            self.model.append(activation_layer())

            for input, output in zip(layer_sizes[0:-1], layer_sizes[1:]):
                self.model.append(nn.Linear(input, output))
                self.model.append(activation_layer())
                self.model.append(nn.Dropout(p=0.3))

            self.model.append(nn.Linear(layer_sizes[-1], self.output_length))
        
        self.model.append(activation_output())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00002, betas=(0.9, 0.999))

    def forward(self, x):
        return self.model.forward(x).view(-1, *self.output_shape)


class Discriminator(nn.Module):
    def __init__(self, input_shape: tuple, layer_sizes: list, activation_layer: nn.Module = nn.ReLU, activation_output: nn.Module = nn.Sigmoid) -> None:
        super(Discriminator, self).__init__()
        self.layer_sizes = layer_sizes
        self.model = nn.Sequential()
        
        self.input_length = np.prod(input_shape)
        self.input_shape = input_shape

        if len(layer_sizes) == 0:
            self.model.append(nn.Linear(self.input_length, 1))
            
        else:
            self.model.append(nn.Linear(self.input_length, layer_sizes[0]))
            self.model.append(activation_layer())

            for input, output in zip(layer_sizes[0:-1], layer_sizes[1:]):
                self.model.append(nn.Linear(input, output))
                self.model.append(activation_layer())
                self.model.append(nn.Dropout(p=0.3))

            self.model.append(nn.Linear(layer_sizes[-1], 1))
        
        self.model.append(activation_output())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00002, betas=(0.9, 0.999))
      

    def forward(self, x):
        x_flat = x.view(-1, self.input_length)
        return self.model.forward(x_flat).view(-1, 1)


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
        self.generator = Generator(100, img_shape, [200, 3000, 10000], activation_output=nn.Tanh)
        self.discriminator = Discriminator(img_shape, [10000, 3000, 200])

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
            self.generator.optimizer.step()
            
            g_loss += gen_loss
            d_loss += discriminator_loss

            if batch_num % 100 == 0:
                print(f'batch {batch_num} gen loss: {gen_loss}, discr loss: {discriminator_loss}')
                plt.imshow((fake_data[0].reshape(self.img_shape).detach().to('cpu').swapaxes(0, 1).swapaxes(1, 2) + 1) / 2)
                plt.show()
        return g_loss, d_loss

    def generator_loss(self, dgz):
        return - torch.mean(clamped_log(dgz))