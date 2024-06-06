from torch import nn
import torchvision


def get_alexnet(n_class):
    model = torchvision.models.alexnet()
    model.classifier[6] = nn.Linear(in_features=4096, out_features=n_class, bias=True)
    return model
