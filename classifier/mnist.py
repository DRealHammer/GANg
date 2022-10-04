import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

class CClassifier(nn.Module):
  
    def __init__(self, input_shape, output_activation: nn.Module = nn.Softmax) -> None:
        super(CClassifier, self).__init__()
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

            nn.Conv2d(128, 256, 4, 2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(slope),

            nn.Conv2d(256, 512, 4, 1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(slope)
            )

        self.lin = nn.Sequential(
            #nn.Dropout(0.1),
            nn.Linear(4*4*512, 10),
            output_activation()
        )

    def forward(self, x):
        x_2d = x.view(-1, self.input_channels, self.input_res, self.input_res)
        conv = self.conv(x_2d).view(-1, 4*4*512)
        return self.lin(conv)


def train_classifier(train_data_loader: DataLoader, n_epochs: int, device):

    classifier = CClassifier((1, 32, 32)).to(device)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))

    loss = lambda probs, labels: torch.nn.functional.nll_loss(torch.log(probs), labels)

    for i in range(n_epochs):
        for batch_num, batch in enumerate(train_data_loader):
            classifier.train(True)
            classifier.zero_grad()
            preds = classifier.forward(batch[0].to(device))
            out = loss(preds.to(device), batch[1].to(device))
            out.backward()
            classifier_optimizer.step()


        print(f"Loss epoch: {i}, loss: {out}")

    correct_num = 0
    for batch in train_data_loader:
        correct = torch.argmax(classifier.forward(batch[0].to(device)), axis=1) == batch[1].to(device)
        correct_num += torch.sum(correct.float())

    print(f'Accuracy of Classifier: {correct_num / len(train_data_loader.dataset)} with {correct_num} / {len(train_data_loader.dataset)} correct')

    examples = 0
    for batch in train_data_loader:
        print('classifier: ', torch.argmax(classifier.forward(batch[0][0].to(device))).item())
        plt.imshow(batch[0][0].view(32, 32, 1))
        plt.show()
        examples += 1
        if examples == 5:
            break
    return classifier

def get_classifier():
    classifier = CClassifier((1, 32, 32))
    classifier.load_state_dict(torch.load('classifier/mnist.pt'))
    classifier.eval()

    classifier = classifier.to('cuda')

    return classifier