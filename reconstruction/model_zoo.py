import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class VAE_mnist(nn.Module):
    def __init__(self):
        super(VAE_mnist, self).__init__()

        self.fc11 = nn.Linear(1024, 400)
        self.fc12 = nn.Linear(400, 400)
        self.fc13 = nn.Linear(400, 200)

        # self.relu1 = nn.ReLU()

        self.fc21 = nn.Linear(200, 20)
        self.fc22 = nn.Linear(200, 20)

        self.fc3 = nn.Linear(20, 200)
        self.fc4 = nn.Linear(200, 400)
        self.fc5 = nn.Linear(400, 1024)

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc13(self.fc12(self.fc11(x))))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = self.fc4(h3)
        x = self.fc5(h3)
        x = x.view(x.size(0), 1, 32, 32)
        return torch.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_relu_feature(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc13(self.fc12(self.fc11(x))))
        return h1

    def get_fc13_feature(self, x):
        x = x.view(x.size(0), -1)
        z = self.fc13(self.fc12(self.fc11(x)))
        return z

    def get_fc12_feature(self, x):
        x = x.view(x.size(0), -1)
        z = self.fc12(self.fc11(x))
        return z


class VAE_cifar(nn.Module):
    def __init__(self):
        super(VAE_cifar, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1,
                               padding=1, bias=True)

        self.fc1 = nn.Linear(1024 * 12, 1200)
        # self.relu1 = nn.ReLU()

        self.fc21 = nn.Linear(1200, 120)
        self.fc22 = nn.Linear(1200, 120)

        self.fc3 = nn.Linear(120, 1200)
        self.fc4 = nn.Linear(1200, 1024 * 12)

        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1,
                               padding=1, bias=True)

    def encode(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = self.fc4(h3)
        x = h3.view(h3.size(0), 12, 32, 32)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return torch.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == '__main__':

    # test mnist VAE model
    test_model = VAE_mnist()
    test_model.cuda()
    x = torch.randn(16, 1, 32, 32).cuda()
    x_vae, mu, logvar = test_model(x)
    print(x_vae.size())

    # test cifar10 VAE model
    test_model = VAE_cifar()
    test_model.cuda()
    x = torch.randn(16, 3, 32, 32).cuda()
    x_vae, mu, logvar = test_model(x)
    print(x_vae.size())