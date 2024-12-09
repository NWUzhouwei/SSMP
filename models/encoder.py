import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from torchvision import models
import timm
from torch.nn import functional as F
class Encoder(torch.nn.Module):
    def __init__(self, ):
        super(Encoder, self).__init__()

        # Layer Definition
        # self.deit = timm.create_model('deit_small_patch16_224', pretrained=True)
        self.resnet = models.resnet50(pretrained=True)
        # self.resnet = torch.nn.Sequential(*[
        #     resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3,
        #     resnet.layer4
        # ])[:6]

        # self.features = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential()
        self.fc = nn.Linear(2048, 256)


    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        x = torch.split(x, 1, dim=0)
        image_features = []

        for img in x:
            features=img.squeeze(dim=0)

            features = self.resnet(features)
            features=self.fc(features)
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            # features = self.layer1(features)
            #
            # # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            # features = self.layer2(features)
            #
            # # print(features.size())    # torch.Size([batch_size, 256, 14, 14])
            # features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 256, 7, 7])
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 7, 7])
        # image_features=image_features.view(-1,image_features.shape[2],image_features.shape[3],image_features.shape[4])
        return image_features
    #1 32 512
    #32 256 56 56








if __name__ == '__main__':
    input_tensor = torch.rand(32,1,3,224,224).cuda()
    model=Encoder().cuda()
    x=model(input_tensor)
    print(x.shape)  #(64,256,256)