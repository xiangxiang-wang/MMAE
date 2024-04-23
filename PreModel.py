##预训练模型，生成mask
import torch
import torch.nn as nn


class PreMask(nn.Module):
    def __init__(self):
        super(PreMask, self).__init__()

        self.process1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.ReLU())

        self.process2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.ReLU())
        # self.process2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True),
        #                               nn.Sigmoid())
        self.patch_size = 8

        self.pool = nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size)
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=self.patch_size)


    def forward(self, x1, x2):
        x1 = self.process1(x1)
        x2 = self.process1(x2)
        output = torch.cat([x1, x2], dim=1)
        output = self.process2(output)
        return output
