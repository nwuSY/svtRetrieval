import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import warnings

warnings.filterwarnings("ignore")


class AttentionModel(nn.Module):
    def __init__(self, hidden_layer=380):
        super(AttentionModel, self).__init__()

        self.attn_hidden_layer = hidden_layer
        self.net = nn.Sequential(
            nn.Conv2d(2048, self.attn_hidden_layer, kernel_size=(1, 1)),
            nn.Conv2d(self.attn_hidden_layer, 1, kernel_size=(1, 1))
        )

    def forward(self, x):
        attn_mask = self.net(x)
        attn_mask = attn_mask.view(attn_mask.size(0), -1)
        attn_mask = nn.Softmax(dim=1)(attn_mask)
        attn_mask = attn_mask.view(attn_mask.size(0), 1, x.size(2), x.size(3))
        x_attn = x * attn_mask
        x = x + x_attn
        return x, attn_mask


class Model(nn.Module):
    def __init__(self, feature_dim=128, embedding_size=604, attention=True):
        super(Model, self).__init__()

        resnet152 = models.resnet152(pretrained=True)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                # print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                # print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)
        self.embedding_size = embedding_size

        self.conv = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2048, 2048, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2048, 2048, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(2048, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        # Semantic Attention Weights
        self.fc_w = nn.Linear(1024, self.embedding_size, bias=False)

        # LAST LAYERS
        self.bn3 = nn.BatchNorm1d(1024 + self.embedding_size)
        # self.fc3 = nn.Linear(1024 + self.embedding_size, num_classes)

        # projection head
        self.g = nn.Sequential(
            nn.Linear(1024 + self.embedding_size, 512, bias=False), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True)
        )

    def forward(self, im, textual_features):
        x = self.cnn_features(im)  # [bz, 2048, 16, 16]

        if self.attention:
            x, attn_mask = self.attn(x)

        x = self.attn_bn(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        visual_features = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc_w(visual_features)

        x = torch.bmm(x.view(im.shape[0], 1, self.embedding_size), textual_features.permute(0, 2, 1))
        x = torch.tanh(x)
        x = F.softmax(x, dim=2)
        # Attention over textual features
        x = torch.bmm(x, textual_features)
        # Fuse
        feature = torch.cat((x[:, 0, :], visual_features), 1)

        out = F.dropout(self.g(feature), p=0.3, training=self.training)

        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
