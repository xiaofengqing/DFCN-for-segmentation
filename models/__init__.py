import torchvision.models as models

from DFCN.models.fcn import *
from DFCN.models.segnet import *
from DFCN.models.unet import *
from DFCN.models.deeplab_vgg import *
from DFCN.models.DFCN import *


def get_model(name, n_classes):
    model = _get_model_instance(name)

    if name in ['fcn32s', 'fcn16s', 'fcn8s']:
        model = model(n_classes=n_classes)
        # initNetParams(model)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == 'segnet':
        model = model(n_classes=n_classes,
                      is_unpooling=True)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == 'unet':
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=3,
                      is_deconv=True)
    elif name == 'deeplab_vgg':
        model = model()
        # model.init_parameters()

    elif name == 'DFCN':
        model = model(n_classes=6)
        initNetParams(model)


    else:
        raise 'Model {} not available'.format(name)

    return model

def _get_model_instance(name):
    return {
        'fcn32s': fcn32s,
        'fcn8s': fcn8s,
        'fcn16s': fcn16s,
        'unet': unet,
        'segnet': segnet,
        'deeplab_vgg':deeplab_vgg,
        'DFCN':DFCN,
    }[name]

def initNetParams(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform(m.weight)
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform(m.weight)
            if m.bias:
                init.constant(m.bias, 0)

