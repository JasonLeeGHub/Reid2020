import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        # load backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)

        # cnn feature
        self.encoder_bkbone = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                         resnet.layer1, resnet.layer2, resnet.layer3)
        self.encoder_shared = resnet.layer4
        self.encoder_rgb = resnet.layer4
        self.encoder_ir = resnet.layer4
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Linear(2048, 3)  # 3types, 0: rgb feature, 1: ir feature, 2, shared feature
    def forward(self, in_rgb, in_ir, in_source):
        fearure_rgb = self.encoder_bkbone(in_rgb)
        feature_shared_rgb = self.encoder_shared(fearure_rgb)
        # feature_shared_rgb = self.GAP(feature_shared_rgb).squeeze()
        feature_private_rgb = self.encoder_rgb(fearure_rgb)
        # feature_private_rgb = self.GAP(feature_private_rgb).squeeze()
        feature_ir = self.encoder_bkbone(in_ir)
        feature_shared_ir = self.encoder_shared(feature_ir)
        # feature_shared_ir = self.GAP(feature_shared_ir).squeeze()
        feature_private_ir = self.encoder_ir(feature_ir)
        # feature_private_ir = self.GAP(feature_private_ir).squeeze()
        feature_source = self.encoder_bkbone(in_source)
        feature_shared_source = self.encoder_shared(feature_source)
        # feature_shared_source = self.GAP(feature_shared_source).squeeze()
        feature_private_source = self.encoder_rgb(feature_source)
        # feature_private_source = self.GAP(feature_private_source).squeeze()


        F_S_R_cls = self.cls(self.GAP(feature_shared_rgb).squeeze())
        F_P_R_cls = self.cls(self.GAP(feature_private_rgb).squeeze())
        F_S_I_cls = self.cls(self.GAP(feature_shared_ir).squeeze())
        F_P_I_cls = self.cls(self.GAP(feature_private_ir).squeeze())
        F_S_S_cls = self.cls(self.GAP(feature_shared_source).squeeze())
        F_P_S_cls = self.cls(self.GAP(feature_private_source).squeeze())

        return [feature_shared_rgb,    feature_private_rgb,    F_S_R_cls, F_P_R_cls], \
               [feature_shared_ir,     feature_private_ir,     F_S_I_cls, F_P_I_cls], \
               [feature_shared_source, feature_private_source, F_S_S_cls, F_P_S_cls]

class Supervize_classifier(nn.Module):
    def __init__(self, class_num):
        super(Supervize_classifier, self).__init__()
        # parameters
        self.class_num = class_num
        # pools
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        dropout = nn.Dropout(p=0.5)

        self.pool_c = nn.Sequential(avgpool, dropout)
        self.classifier = BottleClassifier(2048, self.class_num, relu=True, dropout=False, bottle_dim=256)

    def forward(self,features):
        features = torch.squeeze(self.pool_c(features))
        features = self.classifier(features)
        return features

class embeder (nn.Module):
    def __init__(self):
        super(embeder, self).__init__()
        self.pool_e = nn.AdaptiveAvgPool2d((1, 1))
        self.embeder =  nn.Linear(2048, 256)
    def forward(self, features):
        features = torch.squeeze(self.pool_e(features))
        features = self.embeder(features)
        return features




class Decoder_rgb(nn.Module):
    def __init__(self, conv_dims = 2048):
        super(Decoder_rgb, self).__init__()
        current_dims = conv_dims
        layers = []
        # up sampling layers
        for i in range(5):
            layers.append(nn.ConvTranspose2d(current_dims, current_dims//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(current_dims//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            current_dims = current_dims//2

        # output layer
        layers.append(nn.Conv2d(current_dims, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Decoder_ir(nn.Module):
    def __init__(self, conv_dims=2048):
        super(Decoder_ir, self).__init__()
        current_dims = conv_dims
        layers = []
        # up sampling layers
        for i in range(5):
            layers.append(
                nn.ConvTranspose2d(current_dims, current_dims // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(current_dims // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            current_dims = current_dims // 2

        # output layer
        layers.append(nn.Conv2d(current_dims, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class BottleClassifier(nn.Module):

    def __init__(self, in_dim, out_dim, relu=True, dropout=True, bottle_dim=512):
        super(BottleClassifier, self).__init__()

        bottle = [nn.Linear(in_dim, bottle_dim)]
        bottle += [nn.BatchNorm1d(bottle_dim)]
        if relu:
            bottle += [nn.LeakyReLU(0.1)]
        if dropout:
            bottle += [nn.Dropout(p=0.5)]
        bottle = nn.Sequential(*bottle)
        bottle.apply(weights_init_kaiming)
        self.bottle = bottle

        classifier = [nn.Linear(bottle_dim, out_dim)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.bottle(x)
        x = self.classifier(x)
        return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)