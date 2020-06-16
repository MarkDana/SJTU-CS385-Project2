'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, in_channels=3):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7)) # shape not changed
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.features(x) #512x7x7
        out = self.avgpool(out) # add this if input is 224x224; if input is 32x32, feature out here is 512x1x1
        out = out.view(out.size(0), -1) # equal to torch.flatten(x, 1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)] # use for what?
        return nn.Sequential(*layers)


def VGG11(num_classes=10, in_channels=3):
    return VGG('VGG11', num_classes, in_channels)

def VGG13(num_classes=10, in_channels=3):
    return VGG('VGG13', num_classes, in_channels)

def VGG16(num_classes=10, in_channels=3):
    return VGG('VGG16', num_classes, in_channels)

def VGG19(num_classes=10, in_channels=3):
    return VGG('VGG19', num_classes, in_channels)

if __name__ == '__main__':

    # import torchvision.models as models
    # net = models.vgg16()

    net = VGG16()
    net.train()
    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    for n, m in net.named_modules():
        print(n)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=0., std=0.01)
            m.bias.data.fill_(0.)
    
    net.apply(init_weights)

    for k,v in net.state_dict().items():
        print(k, v.float().mean().item(), v.float().var().item())
    
    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y.size())