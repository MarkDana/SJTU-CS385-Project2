import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from model_zoo import VAE_cifar as VAE
from utils import tensor_to_PILimage
import numpy as np
from PIL import Image


if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    reconstruction_function = nn.MSELoss(size_average=False)
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


def train(args, model, trainloader):
    num_epochs = 50
    learning_rate = 1e-3
    loss_record = []

    optimizer = optim.Adam(model.parameters(), lr=1e-3 * 0.3)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(trainloader):
            img, _ = data
            # img = img.view(img.size(0), -1)
            img = Variable(img)
            if torch.cuda.is_available():
                img = img.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(img)
            loss = loss_function(recon_batch, img, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            loss_record.append(loss.item())
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(img),
                    len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                    loss.item() / len(img)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataloader.dataset)))
        if epoch % 10 == 0:
            save = to_img(recon_batch.cpu().data)
            save_image(save, './vae_img/cifar_image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './vae_cifar10.pth')
    print(loss_record)


def test(model, testloader):
    # model.load_state_dict(torch.load("./vae_cifar10.pth"))
    for batch_idx, data in enumerate(testloader):
        img, targets = data
        img = img.cuda()
        recon_batch, mu, logvar = model(img)
        break
    print(targets)

    z1, mu1, log1 = model(img[5].unsqueeze(dim=0))
    z2, mu2, log2 = model(img[1].unsqueeze(dim=0))
    z3, mu3, log3 = model(img[4].unsqueeze(dim=0))
    z4, mu4, log4 = model(img[0].unsqueeze(dim=0))

    fi_x, delta_x = 0, np.pi / 8 / 2
    fi_y, delta_y = 0, np.pi / 8 / 2
    z = None
    for i in range(9):
        now = None
        print(fi_x / np.pi , fi_y / np.pi)
        for j in range(9):
            if now is None:
                mu = np.cos(fi_x) * (np.cos(fi_y) * mu1 + np.sin(fi_y) * mu2) + \
                      np.sin(fi_x) * (np.cos(fi_y) * mu3 + np.sin(fi_y) * mu4)
                log = np.cos(fi_x) * (np.cos(fi_y) * log1 + np.sin(fi_y) * log2) + \
                      np.sin(fi_x) * (np.cos(fi_y) * log3 + np.sin(fi_y) * log4)

                now = model.reparametrize(mu, log)
                now = model.decode(now)
                now = now.clamp(0, 1)
                now = now[0].cpu().detach().numpy()
                print(now.shape)
            else:
                mu = np.cos(fi_x) * (np.cos(fi_y) * mu1 + np.sin(fi_y) * mu2) + \
                     np.sin(fi_x) * (np.cos(fi_y) * mu3 + np.sin(fi_y) * mu4)
                log = np.cos(fi_x) * (np.cos(fi_y) * log1 + np.sin(fi_y) * log2) + \
                      np.sin(fi_x) * (np.cos(fi_y) * log3 + np.sin(fi_y) * log4)
                new = model.reparametrize(mu, log)
                new = model.decode(new)
                new = new.clamp(0, 1)
                now = np.concatenate((now, new[0].cpu().detach().numpy()), axis=2)
                # print(now[0].size(), now[1].size(), now[2].size())
                # print(now.shape)
            fi_y = fi_y + delta_y

        if z is None:
            z = now
        else:
            z = np.concatenate((z, now), axis=1)
        fi_y = 0
        fi_x = fi_x + delta_x

    res = torch.tensor(z)
    res = tensor_to_PILimage(res)
    res = Image.fromarray(res)
    res.save("./flow_img/CIFAR_FLOW.jpg")


batch_size = 128
img_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
dataset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=img_transform)
# dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


model = VAE()
if False:
    load_path = "./saved_pth/vae_cifar10.pth"
    checkpt = torch.load(load_path)
    model.load_state_dict(checkpt, strict=True)
if torch.cuda.is_available():
    model.cuda()

train(None, model, dataloader)
test(model, testloader)
