# adapted from https://github.com/kuangliu/pytorch-cifar
# python main.py --lr 0.05 -m resnet50 -d Stanford-Dogs -dr 100 -te 400 --resampled
# python main.py --lr 0.01 -m vgg16 -d Stanford-Dogs -dr 100 -te 300 --resampled

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse

from models import *
import dataloader

torch.manual_seed(7) # cpu
torch.cuda.manual_seed(7) #gpu
torch.backends.cudnn.deterministic=True # cudnn

NETWRKS = {'resnet18': ResNet18, 'resnet34': ResNet34, 'resnet50': ResNet50, 'resnet101': ResNet101, 'resnet152': ResNet152,
           'vgg11': VGG11, 'vgg13': VGG13, 'vgg16': VGG16, 'vgg19': VGG19, }

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint') #default false
parser.add_argument('--totalepoch', '-te', default=200, type=int, help='total training epochs')
parser.add_argument('--model', '-m', default='resnet50', type=str, help='resnet18/.../vgg16')
parser.add_argument('--dataset', '-d', default='MNIST', type=str, help='MNIST/CIFAR-10/CUB200/Stanford-Dogs')
parser.add_argument('--finaldampratio', '-dr', default=0, type=int, help='set for exponential lr, linear interpolation, eg. =100')
parser.add_argument('--resampled', '-rs', action='store_true', help='resampled data') #default false
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

image_size = 224 if 'vgg' in args.model else 32
in_channels = 1 if args.dataset == 'MNIST' else 3
batch_size = 32 if 'vgg' in args.model else 128

trainloader = dataloader.loadData(args.dataset, image_size=image_size, istrain=True, resampled=args.resampled, batch_size=batch_size, isshuffle=True, num_workers=2, ispin_memory=True)
testloader = dataloader.loadData(args.dataset, image_size=image_size, istrain=False, resampled=args.resampled, batch_size=batch_size, isshuffle=False, num_workers=2, ispin_memory=True)
num_classes = len(dataloader.SAMPLED_CLS_INDEX[args.dataset]) if args.resampled else len(dataloader.CLASSES[args.dataset])
#eg. num_classes = 100 if resampled else 120, for dogs

# Model
print('==> Building model..')
net = NETWRKS[args.model](num_classes, in_channels)
net = net.to(device)

chpt_dir = f'./checkpoint/{args.model}-{args.dataset}-lr-{args.lr}-dampratio-{args.finaldampratio}{"-resamplednew" if args.resampled else ""}-{args.totalepoch}epochs'
if not os.path.exists(chpt_dir):
    os.mkdir(chpt_dir)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(chpt_dir, 'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=0., std=0.01)
            if m.bias is not None: m.bias.data.fill_(0.)
    net.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if args.finaldampratio > 0:
    one_damp_ratio = (1./args.finaldampratio) ** (1./(args.totalepoch-1))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=one_damp_ratio)
# equal to
# def decreaseLr(final_damp_ratio = 100):
#     one_damp_ratio = (1./final_damp_ratio)**(1./(args.totalepoch-1))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] *= one_damp_ratio

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # if epoch!=0:
    #     decreaseLr()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if args.finaldampratio > 0:
        scheduler.step()

    with open(os.path.join(chpt_dir, 'train_loss.txt'), 'a') as fout:
        fout.write('%d\t%.3f\t%.3f\n' % (epoch, train_loss / len(trainloader), 100. * correct / total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        with open(os.path.join(chpt_dir, 'test_loss.txt'), 'a') as fout:
            fout.write('%d\t%.3f\t%.3f\n' % (epoch, test_loss/len(testloader), 100. * correct / total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(chpt_dir, 'ckpt.pth'))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.totalepoch):
    train(epoch)
    test(epoch)
