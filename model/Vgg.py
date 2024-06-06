from torch import nn
import torchvision


def get_vgg11(n_class):
    model = torchvision.models.vgg11()
    model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=n_class)
    return model


def get_vgg13(n_class):
    model = torchvision.models.vgg13()
    model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=n_class)
    return model


def get_vgg16(n_class):
    model = torchvision.models.vgg16()
    model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=n_class)
    return model


def get_vgg19(n_class):
    model = torchvision.models.vgg19()
    model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=n_class)
    return model


def get_vgg11_bn(n_class):
    model = torchvision.models.vgg11_bn()
    model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=n_class)
    return model


def get_vgg13_bn(n_class):
    model = torchvision.models.vgg13_bn()
    model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=n_class)
    return model


def get_vgg16_bn(n_class):
    model = torchvision.models.vgg16_bn()
    model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=n_class)
    return model


def get_vgg19_bn(n_class):
    model = torchvision.models.vgg19_bn()
    model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=n_class)
    return model

