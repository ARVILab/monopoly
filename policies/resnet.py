import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, downsample=None):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.bn(out)
        out = F.relu(out)
        # out = self.fc(out)
        # out = F.relu(out)
        # out = self.bn(out)

        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, obs_shape, output_size, hidden_size=128, layers=[2, 2, 2]):
        super(ResNet, self).__init__()

        self.n_inputs = obs_shape
        self._hidden_size = hidden_size
        block = ResidualBlock

        self.fc = nn.Linear(self.n_inputs, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.layer1 = self.make_layer(block, hidden_size, layers[0])
        self.layer2 = self.make_layer(block, hidden_size, layers[1], True)
        self.layer3 = self.make_layer(block, hidden_size, layers[2], True)

        self.pi = nn.Linear(hidden_size, output_size)
        self.v = nn.Linear(hidden_size, 1)


    def make_layer(self, block, hidden_size, blocks, use_downsample=False):
        downsample = None
        if use_downsample:
            downsample = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size)
            )
        layers = []
        layers.append(block(hidden_size, downsample))
        for i in range(1, blocks):
            layers.append(block(hidden_size))

        return nn.Sequential(*layers)

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        # out = self.layer3(out)

        action_features = self.pi(out)
        value = self.v(out)

        return value, action_features
