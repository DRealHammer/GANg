import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import gc

def create_grid(X):
    res = int(X.shape[0]**(1/2))
    channels = X.shape[1]
    img_res = X.shape[2]
    if channels == 3:
        return X.cpu().detach().numpy().swapaxes(1, 2).swapaxes(2, 3).reshape(res, res, img_res, img_res, channels).swapaxes(1, 2).reshape(res*img_res, res*img_res, channels)
    
    if channels == 1:
        return X.cpu().detach().numpy().swapaxes(1, 2).swapaxes(2, 3).reshape(res, res, img_res, img_res, channels).swapaxes(1, 2).reshape(res*img_res, res*img_res)


def clamped_log(x, eps=1e-16):
    return torch.clamp(torch.log(torch.clamp(x, eps)), -100)



def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.01)
        nn.init.constant_(m.bias.data, 0)


class GANScore():
    def __init__(self) -> None:

        # marginal of GAN
        self.py = None

        # marginal of dataset
        self.py_squiggly = None

        # posterior of GAN
        self.pyx = None

    
    def kl_divergence(self, px, qx, eps=1e-16, dim=1):
        return torch.sum(px * (clamped_log(px) - clamped_log(qx)), dim=dim)


    def calc_marginal_gan(self, classifier, gan, z, runs, img_transform=transforms.Compose([])):
        self.py = torch.zeros((10)).to("cuda")

        
        for i in range(runs):
            zi_cuda = z[i].to('cuda')
            z_gen = gan.generator.forward(zi_cuda)
            del zi_cuda
            z_transformed = img_transform(z_gen)
            del z_gen
            #pyz = classifier(z_transformed)[0]
            pyz = classifier(z_transformed)
            del z_transformed
            pyz = pyz.detach()

            py_temp = torch.mean(pyz, dim=0)
            self.py += py_temp

            gc.collect()
            
        self.py /= runs
        return self.py

    def calc_marginal_dataset(self, dataset):
        y = dataset.targets
        counts = torch.bincount(torch.Tensor(y).int(), minlength=len(dataset.classes))
        self.py_squiggly = (counts / len(y)).to('cuda')

    def calc_posterior(self, classifier, gan, z, runs, img_transform):
        self.pyx = torch.Tensor([]).to('cuda')
        for i in range(runs):
            zi_cuda = z[i].to('cuda')
            z_gen = gan.generator.forward(zi_cuda)
            del zi_cuda
            z_transformed = img_transform(z_gen)
            del z_gen
            self.pyx = torch.cat((self.pyx, classifier(z_transformed)))
            del z_transformed
            gc.collect()
            if i % int((runs / 3)) == 0:
                torch.cuda.empty_cache()

    def inception(self):
        divergence = self.kl_divergence(self.pyx, self.py)
        return torch.exp((torch.mean(divergence)))

    def mode_score(self):
        return torch.exp((torch.mean(self.kl_divergence(self.pyx, self.py_squiggly))) - self.kl_divergence(self.py, self.py_squiggly, dim=0))
