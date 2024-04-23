# 复杂度测试
import torch
# from torchstat import stat
from thop import profile
from data_loader import DataTest
import argparse
from modules import MMAE
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import time
# import matplotlib.pyplot as plt
# import numpy as np
#
# from PIL import Image
# import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=64)
    parser.add_argument("--n_resblocks", type=int, default=3)
    parser.add_argument("--n_convs", type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--testData_dir', type=str, default="DataSet/testData/lytro")
    # parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\lytro2")
    # parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\lytro3")
    # parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\MFFW_select3")
    # parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\lytro_select3")
    # parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\MFI_WHU_select3")
    # parser.add_argument('--testData_dir', type=str, default="D:/DeepLearing/set/MFFW")
    # parser.add_argument('--saveModel_dir', type=str, default='E:\MMAE\save_model\epoch_82_loss_6.166936.pth')
    # parser.add_argument('--result', type=str, default=r'D:\BaiduSyncdisk\MMAE\result/')

    return parser.parse_args()


args = parse_args()
net = MMAE(args)
# input1, input2 = torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)
# macs, params = profile(net, inputs=(input1, input2,))
# name = 'MMAE'
# print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
# print("---|---|---")
# print("%s | %.2f | %.2f" % (name, params / (1000 ** 2), macs / (1000 ** 3)))
# input1, input2 = torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)
# input3 = torch.rand(1, 1, 16, 16)
# macs, params = profile(net.mae, inputs=(input1, input2,input3))
# name = 'MAE'
# print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
# print("---|---|---")
# print("%s | %.2f | %.2f" % (name, params / (1000 ** 2), macs / (1000 ** 2)))
input1, input2 = torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)
input3 = torch.rand(1, 256, 768)
macs, params = profile(net.mae.patch_attention, inputs=(input3,16,16))
name = 'patchattention'
print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
print("---|---|---")
print("%s | %.2f | %.2f" % (name, params / (1000 ** 2), macs / (1000 ** 2)))