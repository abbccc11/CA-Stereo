import torch

from submodule import *
import math
import gc
import time
class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(1, 16, 5, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(16, 32, 5, 2, 1, 1),
                                       nn.ReLU(inplace=True)
                                       )

        self.layer1 = self._make_layer(BasicBlock, 32, 4, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 32, 2, 1, 1, 2)
        self.layer3 = self._make_layer(BasicBlock, 32, 2, 1, 1, 4)
        self.layer4 = self._make_layer(BasicBlock, 32, 2, 1, 1, 1)
        self.branch1 = nn.Sequential(SoftPool2d(1),
                                     convbn(32, 16, 1, 1, 0, 1)
                                     )

        self.branch2 = nn.Sequential(SoftPool2d(2),
                                     convbn(32, 16, 1, 1, 0, 1)
                                     )

        self.branch3 = nn.Sequential(SoftPool2d(4),
                                     convbn(32, 16, 1, 1, 0, 1)
                                     )

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)

        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        x0 = self.branch1(output_skip)
        x1 = self.branch2(output_skip)
        x2 = self.branch3(output_skip)

        return [x0, x1, x2]


