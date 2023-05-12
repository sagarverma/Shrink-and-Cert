import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import GetSubnetUnstructured


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# lin_i: i layer linear feedforard network.
def lin_1(input_dim=3072, num_classes=10):
    model = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, num_classes))
    return model


def lin_2(input_dim=3072, hidden_dim=100, num_classes=10):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.Linear(hidden_dim, num_classes),
    )
    return model


def lin_3(input_dim=3072, hidden_dim=100, num_classes=10):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Linear(hidden_dim, num_classes),
    )
    return model


def lin_4(input_dim=3072, hidden_dim=100, num_classes=10):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Linear(hidden_dim, num_classes),
    )
    return model


def mnist_model(conv_layer, linear_layer, **kwargs):
    model = nn.Sequential(
        conv_layer(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        conv_layer(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        linear_layer(32 * 7 * 7, 100),
        nn.ReLU(),
        linear_layer(100, 10),
    )
    return model


def mnist_model_large(conv_layer, linear_layer, **kwargs):
    model = nn.Sequential(
        conv_layer(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        conv_layer(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        conv_layer(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        conv_layer(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        linear_layer(64 * 7 * 7, 512),
        nn.ReLU(),
        linear_layer(512, 512),
        nn.ReLU(),
        linear_layer(512, 10),
    )
    return model


def cifar_model(conv_layer, linear_layer, **kwargs):
    model = nn.Sequential(
        conv_layer(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        conv_layer(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        linear_layer(32 * 8 * 8, 100),
        nn.ReLU(),
        linear_layer(100, 10),
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            m.bias.data.zero_()
    return model

class CIFAR_Model_Large(nn.Module):
    def __init__(self, conv_layer, linear_layer, num_classes=10, k=1.0, unstructured=True):
        super(CIFAR_Model_Large, self).__init__()
        self.k = k 
        self.unstructured_pruning = unstructured

        self.conv1 = conv_layer(3, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = conv_layer(32, 32, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = conv_layer(32, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = conv_layer(64, 64, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.linear1 = linear_layer(64 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.linear2 =  linear_layer(512, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.linear3 = linear_layer(512, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2.0 / n))
        #         m.bias.data.zero_()

    def forward(self, x):
        if self.unstructured_pruning:
            score_list = []
            for (name, vec) in self.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        score_list.append(attr.view(-1))
            scores = torch.cat(score_list)
            adj = GetSubnetUnstructured.apply(scores.abs(), self.k)

            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            numel = attr.numel()
                            vec.w = attr * adj[pointer: pointer + numel].view_as(attr)
                            pointer += numel

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn5(self.linear1(out)))
        out = F.relu(self.bn6(self.linear2(out)))
        out = self.linear3(out)
        return out

def cifar_model_large(conv_layer, linear_layer, **kwargs):
    return CIFAR_Model_Large(conv_layer, linear_layer, **kwargs)


class DenseSequential(nn.Sequential):
    def forward(self, x):
        xs = [x]
        for module in self._modules.values():
            if "Dense" in type(module).__name__:
                xs.append(module(*xs))
            else:
                xs.append(module(xs[-1]))
        return xs[-1]


class Dense(nn.Module):
    def __init__(self, *Ws):
        super(Dense, self).__init__()
        self.Ws = nn.ModuleList(list(Ws))
        if len(Ws) > 0 and hasattr(Ws[0], "out_features"):
            self.out_features = Ws[0].out_features

    def forward(self, *xs):
        xs = xs[-len(self.Ws):]
        out = sum(W(x) for x, W in zip(xs, self.Ws) if W is not None)
        return out


def cifar_model_resnet(conv_layer, linear_layer, N=5, factor=1, **kwargs):
    def block(in_filters, out_filters, k, downsample):
        if not downsample:
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else:
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(
                conv_layer(
                    in_filters, out_filters, k_first, stride=skip_stride, padding=1
                )
            ),
            nn.ReLU(),
            Dense(
                conv_layer(
                    in_filters, out_filters, k_skip, stride=skip_stride, padding=0
                ),
                None,
                conv_layer(out_filters, out_filters, k, stride=1, padding=1),
            ),
            nn.ReLU(),
        ]

    conv1 = [conv_layer(3, 16, 3, stride=1, padding=1), nn.ReLU()]
    conv2 = block(16, 16 * factor, 3, False)
    for _ in range(N):
        conv2.extend(block(16 * factor, 16 * factor, 3, False))
    conv3 = block(16 * factor, 32 * factor, 3, True)
    for _ in range(N - 1):
        conv3.extend(block(32 * factor, 32 * factor, 3, False))
    conv4 = block(32 * factor, 64 * factor, 3, True)
    for _ in range(N - 1):
        conv4.extend(block(64 * factor, 64 * factor, 3, False))
    layers = (
            conv1
            + conv2
            + conv3
            + conv4
            + [
                Flatten(),
                linear_layer(64 * factor * 8 * 8, 1000),
                nn.ReLU(),
                linear_layer(1000, 10),
            ]
    )
    model = DenseSequential(*layers)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
    return model


def vgg4_without_maxpool(conv_layer, linear_layer, **kwargs):
    model = nn.Sequential(
        conv_layer(3, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        conv_layer(64, 64, 3, stride=2, padding=1),
        nn.ReLU(),
        conv_layer(64, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        conv_layer(128, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        linear_layer(128 * 8 * 8, 256),
        nn.ReLU(),
        linear_layer(256, 256),
        nn.ReLU(),
        linear_layer(256, 10),
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            m.bias.data.zero_()
    return model


def cifar_model_resnet(N=5, factor=10):
    def block(in_filters, out_filters, k, downsample):
        if not downsample:
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else:
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(
                nn.Conv2d(
                    in_filters, out_filters, k_first, stride=skip_stride, padding=1
                )
            ),
            nn.ReLU(),
            Dense(
                nn.Conv2d(
                    in_filters, out_filters, k_skip, stride=skip_stride, padding=0
                ),
                None,
                nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1),
            ),
            nn.ReLU(),
        ]

    conv1 = [nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU()]
    conv2 = block(16, 16 * factor, 3, False)
    for _ in range(N):
        conv2.extend(block(16 * factor, 16 * factor, 3, False))
    conv3 = block(16 * factor, 32 * factor, 3, True)
    for _ in range(N - 1):
        conv3.extend(block(32 * factor, 32 * factor, 3, False))
    conv4 = block(32 * factor, 64 * factor, 3, True)
    for _ in range(N - 1):
        conv4.extend(block(64 * factor, 64 * factor, 3, False))
    layers = (
            conv1
            + conv2
            + conv3
            + conv4
            + [
                Flatten(),
                nn.Linear(64 * factor * 8 * 8, 1000),
                nn.ReLU(),
                nn.Linear(1000, 10),
            ]
    )
    model = DenseSequential(*layers)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
    return model
