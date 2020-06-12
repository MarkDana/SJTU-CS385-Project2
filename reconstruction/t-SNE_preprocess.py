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
from model_zoo import VAE_mnist as VAE
from utils import tensor_to_PILimage
import numpy as np
from PIL import Image


if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 32, 32)
    return x


def get_encoder_feature(model, testloader, save_path):
    # model.load_state_dict(torch.load("./vae_cifar10.pth"))
    feature_list = None
    label_list = None
    for batch_idx, data in enumerate(testloader):
        img, targets = data
        img = img.cuda()
        encoder_feature = model.get_fc12_feature(img)
        if feature_list is None:
            feature_list = encoder_feature.detach().cpu()
            label_list = targets
        else:
            feature_list = torch.cat((feature_list, encoder_feature.detach().cpu()), dim=0)
            label_list = torch.cat((label_list, targets), dim=0)

    feature_list = feature_list.numpy()
    label_list = label_list.numpy()
    print("in total feature numbers and feature dimension ", feature_list.shape)
    print("in total label numbers ", label_list.shape)
    np.save(os.path.join(save_path, "mnist_encoder_fc12_feature.npy"), feature_list, allow_pickle=True)
    np.save(os.path.join(save_path, "mnist_label.npy"), label_list, allow_pickle=True)


batch_size = 128
img_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
dataset = MNIST('./data', transform=img_transform, download=True, train=False)
testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


model = VAE()
load_path = "./saved_pth/vae_mnist.pth"
checkpt = torch.load(load_path)
model.load_state_dict(checkpt, strict=True)
if torch.cuda.is_available():
    model.cuda()

save_path = "./t-SNE_feature"
if not os.path.exists(save_path):
    os.mkdir(save_path)
get_encoder_feature(model, testloader, save_path)

