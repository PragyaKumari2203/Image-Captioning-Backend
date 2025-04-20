import torch
import torch.nn as nn
from torchvision import models

class VGG16Encoder(nn.Module):
    def __init__(self):
        super(VGG16Encoder, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc = nn.Sequential(*list(vgg.classifier.children())[:-1])
        for params in self.parameters():
            params.requires_grad = False

    def forward(self, images):
        x = self.features(images)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x