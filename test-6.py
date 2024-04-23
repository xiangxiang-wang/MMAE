#医学测试图像
import torch
from data_loader import DataTest
import argparse
from modules4 import MMAE
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import os

from utils import test_save_images, test_save_images2


def test(test_dataloader, args):
    device = args.device
    # set_dir = args.testData_dir
    # set_list = os.listdir(set_dir)
    # 获取文件名，去掉后缀
    file_list = os.listdir(args.testData_dir)
    temp_dir = os.listdir(os.path.join(args.testData_dir, file_list[0]))
    set_list = []
    for i in temp_dir:
        portion = os.path.splitext(i)  # 把文件名拆分为名字和后缀

        set_list.append(portion[0])
    print(set_list)
    i = 0
    net = MMAE(args).to(device)
    checkpoint = torch.load(args.saveModel_dir)
    net.load_state_dict(checkpoint['state_dict'], strict=False)
    # pretrained_checkpoint_path = r'E:\MMAE\premask\epoch_140_loss_2.950339.pth'
    # pretrained_net = torch.load(pretrained_checkpoint_path)
    # net.load_state_dict(pretrained_net['state_dict'], strict=False)
    # net.load_state_dict(pretrained_net['net'], strict=False)
    # print("加载预训练模型成功")
    net.eval()
    print(net)
    t1 = time.time()
    for i_test, (image1, image2) in enumerate(test_dataloader):
        image1 = image1.to(device)
        image2 = image2.to(device)
        out, pre_mask = net(image1, image2)
        # 1.2 2
        # out1 = out + 1.2 * (image1 - image2)
        # out2 = out + 2 * (image2 - image1)
        # out1 = out + 2 * (image1 - image2)
        # out2 = out + 2 * (image2 - image1)

        # out1 = out + 2 * (image1 - image2)
        # out2 = out + 2.3 * (image2 - image1)
        out1 = out + 0.5 * (image1 - image2) #1 0.5 1
        out2 = out + 1 * (image2 - image1) #1.3 1 1.3




        # out1 = out1.clamp(0, 1)
        # out2 = out2.clamp(0, 1)
        # out2 = out1 + 1.7 * (image1 - image2)
        # out = out1 + out2
        # out2 = torch.max(out2, image2)
        # out1 = torch.mean(out1, dim=1, keepdim=True)
        # out2 = torch.mean(out, dim=1, keepdim=True)
        # out2 = image2 + 1.7 * (image2 - out)
        # out2 = 1 * out
        # out = out2
        # out2 = 2 * torch.max(out,image2)
        # out = torch.abs(image2 - image1)
        # out = image1 - image2
        # out = 0.8*image1 + 0.2*image2
        # out = torch.max(image1,image2)
        # out, _ = net(out1, image2)
        # out1 = out1 * image2 + out2 *image1
        # out1 = out1 * image2
        # out2 = out
        # out1, _ = net.ch_att(out1,image2)
        # out, pre_mask = net(out1, image2)
        image1_0 = image1[:, 0, :, :]
        image1_1 = image1[:, 1, :, :]
        image1_2 = image1[:, 2, :, :]
        image1_0 = torch.unsqueeze(image1_0, dim=1)
        image1_1 = torch.unsqueeze(image1_1, dim=1)
        image1_2 = torch.unsqueeze(image1_2, dim=1)
        # # x = pre_mask.shape
        # image1_ = torch.cat([image1_0,image1_1,image1_2],dim=1)
        # out = image1_
        image1_c1 = torch.cat([image1_0, image1_0, image1_0], dim=1)
        image_temp_c1 = image2 - image1_c1
        s1 = torch.ones_like(image_temp_c1)
        s2 = torch.zeros_like(image_temp_c1)
        # # s3 = image_temp.clone()
        # image_temp_c1 = torch.where(image_temp_c1 >= 0, image_temp_c1, s2)
        # image_temp_c1 = torch.where(image_temp_c1 <= 1, image_temp_c1, s1)
        image_temp_c1 = image_temp_c1.clamp_(0, 1)
        # image_temp_c1 = torch.abs(image_temp_c1)

        image1_c2 = torch.cat([image1_1, image1_1, image1_1], dim=1)
        image_temp_c2 = image2 - image1_c2
        # s1 = torch.ones_like(image_temp_c2)
        # s2 = torch.zeros_like(image_temp_c2)
        # s3 = image_temp.clone()
        # image_temp_c2 = torch.where(image_temp_c2 >= 0, image_temp_c2, s2)
        # image_temp_c2 = torch.where(image_temp_c2 <= 1, image_temp_c2, s1)
        image_temp_c2 = image_temp_c2.clamp_(0, 1)
        # image_temp_c2 = torch.abs(image_temp_c2)

        image1_c3 = torch.cat([image1_2, image1_2, image1_2], dim=1)
        image_temp_c3 = image2 - image1_c3
        # s1 = torch.ones_like(image_temp)
        # s2 = torch.zeros_like(image_temp)
        # s3 = image_temp.clone()
        # image_temp_c3 = torch.where(image_temp_c3 >= 0, image_temp_c3, s2)
        # image_temp_c3 = torch.where(image_temp_c3 <= 1, image_temp_c3, s1)
        image_temp_c3 = image_temp_c3.clamp_(0, 1)
        # image_temp_c3 = torch.abs(image_temp_c3)

        # image2_0 = image2[:, 0, :, :]
        # image2_1 = image2[:, 1, :, :]
        # image2_2 = image2[:, 2, :, :]
        # image2_0 = torch.unsqueeze(image2_0, dim=1)
        # image2_1 = torch.unsqueeze(image2_1, dim=1)
        # image2_2 = torch.unsqueeze(image2_2, dim=1)
        # image_temp_0 = image2_0 - image1_0
        # image_temp_1 = image2_1 - image1_1
        # image_temp_2 = image2_2 - image1_2
        # x = pre_mask.shape
        # image2_ = torch.cat([image_temp_0, image_temp_1, image_temp_2], dim=1)
        # out = image2_ + out
        # out = net.sigmoid(out)
        # image1 = torch.cat([image1_0, image1_1, image1_2], dim=1)
        # image2 = torch.cat([image2_0, image2_1, image2_2], dim=1)
        # out, pre_mask = net(out1, 1.5 * out2)
        #
        # out = torch.abs(out)
        # out = image1-image2

        #
        # EPSILON = 1e-10
        # mask1 = torch.exp(out1) / (torch.exp(out2) + torch.exp(out1) + EPSILON)
        # mask2 = torch.exp(out2) / (torch.exp(out1) + torch.exp(out2) + EPSILON)
        # out1 = mask1 * out1
        # # # # # # # # sampled_images1 = mask1 * image1
        # # # # # # # # #
        # out2 = mask2 * out2
        # out = out2
        # #
        # out2 = 1.5 * image2
        # out1 = out1*out2
        # out_premask = (image_temp_c1+image_temp_c2+image_temp_c3)/3
        out_premask = torch.min(image_temp_c1, image_temp_c2)
        out_premask = torch.min(out_premask, image_temp_c3)
        # out_premask = torch.max(image_temp_c1,image_temp_c2)
        # out_premask =torch.max(out_premask,image_temp_c3)
        # out_premask = image_temp_c1
        # out_premask = torch.where(out_premask > 0, s1, s2)
        out_premask = torch.abs(out_premask)
        # out1 = torch.where(out_premask > 0, out1, s2)
        # out2 = torch.where(out_premask > 0, out2, s2)
        # out_premask = torch.where(out_premask > 0, out_premask, s2)
        # out_premask = out_premask.clamp(0,1)


        # EPSILON = 1e-10
        # mask1 = torch.exp(image1) / (torch.exp(image2) + torch.exp(image1) + EPSILON)
        # mask2 = torch.exp(image2) / (torch.exp(image1) + torch.exp(image2) + EPSILON)
        # # # # # # # # # # # # # print(mask1)
        # # # # # # # # # # # # # # print(mask2)
        # out1 = mask1 * out1
        # # # # # # # sampled_images1 = mask1 * image1
        # # # # # # # #
        # out2 = mask2 * out2
        # out = out2
        #
        # EPSILON = 1e-10
        # mask1 = torch.exp(out1) / (torch.exp(out2) + torch.exp(out1) + EPSILON)
        # mask2 = torch.exp(out2) / (torch.exp(out1) + torch.exp(out2) + EPSILON)
        # out1 = mask1 * out1
        # # # # # # # sampled_images1 = mask1 * image1
        # # # # # # # #
        # out2 = mask2 * out2
        #
        # out = torch.max(out1,out2)
        # out_ = out1 + out2
        # out_premask = (image_temp_c1+image_temp_c2+image_temp_c3)/3
        # out_premask = torch.where(out_premask > 0.5, s1, s2)
        # out1 = out1.clamp(0, 1)
        # out2 = out2.clamp(0, 1)
        # out1 = torch.where(out_premask > 0, out1, s2)
        # out2 = torch.where(out_premask > 0, out2, s2)
        out = (1 - out_premask) * out1 + out_premask * out2
        # out = out_premask
        # out = torch.max(image1,image2)

        # out = (out1 + out2)/2
        # out =out1
        out = torch.squeeze(out)
        test_save_images(out, args.result + set_list[i] + ".png")
        print(args.result + set_list[i] + ".png")
        i = i + 1
    t2 = time.time()
    print(t2 - t1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=64)
    parser.add_argument("--n_resblocks", type=int, default=3)
    parser.add_argument("--n_convs", type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--testData_dir', type=str, default="DataSet/testData/lytro")
    # parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\lytro2")
    # parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\Roadscene_select3")
    # parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\TNO_select3")
    # parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\MSRS_select3")
    # parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\Harvard_select3")
    parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\GFP_select3")
    # parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\Histological_select3")

    # parser.add_argument('--testData_dir', type=str, default="D:/DeepLearing/set/MFFW")
    # parser.add_argument('--saveModel_dir', type=str, default='E:\MMAE\save_model_cvc14\epoch_156_loss_0.115862.pth')
    # parser.add_argument('--saveModel_dir', type=str, default='E:\MMAE\save_model_harvard\epoch_398_loss_0.142021.pth')
    parser.add_argument('--saveModel_dir', type=str, default='E:\MMAE\save_model\epoch_82_loss_6.166936.pth')
    parser.add_argument('--result', type=str, default=r'D:\BaiduSyncdisk\MMAE\result/')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # transforms_ = [transforms.Resize(size=(args.image_size_h, args.image_size_w)), transforms.ToTensor()]
    # transforms_ = [transforms.Resize(size=(64, 64)), transforms.ToTensor()]
    transforms_ = [transforms.ToTensor()]
    test_set = DataTest(testData_dir=args.testData_dir, transforms_=transforms_)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    test(test_dataloader, args)
