import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, indim, outdim, stride, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(indim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(outdim)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.dropout = dropout
        self.equaldim = (indim == outdim)
        
        if not self.equaldim:
            self.shortcut = nn.Conv2d(indim, outdim, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = None
    
    def forward(self, x):
        if self.equaldim:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
            out = F.dropout(out, p=self.dropout, training=self.training)
            out = self.conv2(out)
            return torch.add(x, out)
        else:
            x = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(x)))
            out = F.dropout(out, p=self.dropout, training=self.training)
            out = self.conv2(out)
            return torch.add(self.shortcut(x), out)


class ResidualBlock(nn.Module):
    def __init__(self, depth, indim, outdim, stride, dropout=0.0):
        super(ResidualBlock, self).__init__()
        layers = []
        layers.append(BasicBlock(indim, outdim, stride))
        for i in range(1, int(depth)):
            layers.append(BasicBlock(outdim, outdim, 1, dropout))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WRN(nn.Module):
    def __init__(self, depth, num_classes, k=10, dropout=0.0):
        super(WResNet, self).__init__()
        dims = [16, 16*k, 32*k, 64*k]
        n = (depth - 4) // 6
        self.outdim = dims[3]

        self.conv = nn.Conv2d(3, dims[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        self.r1 = ResidualBlock(n, dims[0], dims[1], 1, dropout)
        self.r2 = ResidualBlock(n, dims[1], dims[2], 2, dropout)
        self.r3 = ResidualBlock(n, dims[2], dims[3], 2, dropout)

        self.bn1 = nn.BatchNorm2d(dims[3])
        self.relu = nn.ReLU(inplace=True)
        
        self.fc = nn.Linear(dims[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        out = self.r1(out)
        out = self.r2(out)
        out = self.r3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.outdim)
        return self.fc(out)
