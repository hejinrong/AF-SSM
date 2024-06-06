import torchvision
from torch import nn


def get_resnet18(n_class):
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, n_class)
    return model


def get_resnet34(n_class):
    model = torchvision.models.resnet34()
    model.fc = nn.Linear(model.fc.in_features, n_class)
    return model


def get_resnet50(n_class):
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, n_class)
    return model

def get_resnet101(n_class):
    model = torchvision.models.resnet101()
    model.fc = nn.Linear(model.fc.in_features, n_class)
    return model

def get_resnet152(n_class):
    model = torchvision.models.resnet152()
    model.fc = nn.Linear(model.fc.in_features, n_class)
    return model

