import torch
import torchvision
from thop import profile
from tf_bodypix.api import load_model
import Model

print('==> Building model..')

net = 'MobilNet'
# net = 't2t'
# net = 'resnet18'
# net = 'resnet101'
# net = 'cait'
# net = 'GoogLeNet'
model = Model.get_net(net)


input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, (input,))
print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))