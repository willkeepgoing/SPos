import numpy as np
import torch
import torchvision
from torch import nn, optim
from config import config
import math
import torch.utils.model_zoo as model_zoo
import os
import torch.nn.functional as F
from mlit import MliT
import timm

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def l2_norm(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 自己搭建一个简单卷积神经网络
class myModel(nn.Module):
    def __init__(self, num_classes):
        super(myModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3),  # in_channels out_channels kernel_size
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 149
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2),  # 74    #
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 37

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2),  # 18
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 9
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2592, 120),
            nn.ReLU(True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet18, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

        # self.fc2 = nn.Linear(512, num_classes)

        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()

        self.ca1 = ChannelAttention(512)
        self.sa1 = SpatialAttention()

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        # x = self.ca(x) * x
        # x = self.sa(x) * x

        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # x = self.ca1(x) * x
        # x = self.sa1(x) * x

        x = self.backbone.avgpool(x)

        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResNet101(nn.Module):
    def __init__(self, model, num_classes=4):
        super(ResNet101, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(2048, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)
        # self.prelu = nn.PReLU()

        # attention
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()

        self.ca1 = ChannelAttention(2048)
        self.sa1 = SpatialAttention()

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        # x = self.prelu(x)

        # x = self.ca(x) * x
        # x = self.sa(x) * x

        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # x = self.ca1(x) * x
        # x = self.sa1(x) * x

        x = self.backbone.avgpool(x)

        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResNet20(nn.Module):

    def __init__(self, in_channel=3, num_classes=4):
        super(ResNet20, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(2, 2)
        self.block1 = self._make_layer(16, 16, 3)
        self.block2 = self._make_layer(16, 32, 3, downsample=True)
        self.block3 = self._make_layer(32, 64, 3, downsample=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channel, out_channel, num_block, downsample=False):
        layers = []
        if downsample:
            downsample = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False),
                                       nn.BatchNorm2d(out_channel))
        else:
            downsample = None

        layers.append(ResNetBlock_new(in_channel, out_channel, downsample))
        for i in range(1, num_block):
            layers.append(ResNetBlock_new(out_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.view((x.shape[0], -1))
        x = self.fc(x)
        return x


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=4):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = x.view(-1, 512 * 7 * 7)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(pretrained=False, model_root=None, **kwargs):
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19'], model_root))
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
    return model


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)  # N x 64 x 112 x 112
        x = self.maxpool1(x)  # N x 64 x 56 x 56

        x = self.conv2(x)  # N x 64 x 56 x 56
        x = self.conv3(x)  # N x 192 x 56 x 56
        x = self.maxpool2(x)  # N x 192 x 28 x 28

        x = self.inception3a(x)  # N x 256 x 28 x 28
        x = self.inception3b(x)  # N x 480 x 28 x 28
        x = self.maxpool3(x)  # N x 480 x 14 x 14

        x = self.inception4a(x)  # N x 512 x 14 x 14
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4b(x)  # N x 512 x 14 x 14
        x = self.inception4c(x)  # N x 512 x 14 x 14
        x = self.inception4d(x)  # N x 528 x 14 x 14
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)  # N x 832 x 14 x 14
        x = self.maxpool4(x)  # N x 832 x 7 x 7

        x = self.inception5a(x)  # N x 832 x 7 x 7
        x = self.inception5b(x)  # N x 1024 x 7 x 7

        x = self.avgpool(x)  # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)  # N x 1024
        x = self.dropout(x)
        x = self.fc(x)  # N x 1000 (num_classes)

        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # Connect the four branches together
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


# Auxiliary classifier: class InceptionAux, including avetool+conv+fc1+fc2
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)  # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)  # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)  # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)  # N x 1024
        x = self.fc2(x)  # N x num_classes
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


def get_net(choosen):
    if choosen == 'vgg16':
        return vgg16(False, None)
    elif choosen == 'resnet18':
        backbone = torchvision.models.resnet18(pretrained=True)
        models = ResNet18(backbone, config.num_classes)
        return models
    elif choosen == 'resnet101':
        backbone = torchvision.models.resnet101(pretrained=True)
        models = ResNet101(backbone, config.num_classes)
        return models
    elif choosen == 'GoogLeNet':
        models = GoogLeNet()
        return models
    elif choosen == 'VIT':
        models = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=config.num_classes)
        # models = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=config.num_classes)
        # models = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=config.num_classes)
        # models = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=config.num_classes)
        return models
    elif choosen == 'MliT_half':
        models = MliT(img_size=224, patch_size=16, num_classes=config.num_classes, dim=1024,
                     mlp_dim_ratio=2, depth=6, heads=30, dim_head=1024 // 40,  # 24
                     dropout=0., emb_dropout=0.,
                     is_SDC=True, is_LT=True)
        return models
    elif choosen == 'MliT_whole':
        models = MliT(img_size=224, patch_size=16, num_classes=config.num_classes, dim=1024,
                     mlp_dim_ratio=2, depth=6, heads=30, dim_head=1024 // 30,  # 24
                     dropout=0., emb_dropout=0.,
                     is_SDC=True, is_LT=True)
        return models

# print(get_net())
