####带断点恢复
import os

from torchvision.transforms import InterpolationMode

from PreModel import PreMask
import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from Pre_data_loader import SalObjDataset
from Loss import LpLssimLoss
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.nn import init
import numpy as np
import random
import time







def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)


def training_setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 在需要生成随机数据的实验中，每次实验都需要生成数据。
    # 设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。使得每次运行该 .py 文件时生成的随机数相同。

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(train_loader, args):
    device = args.device
    n_epochs = args.n_epochs
    model_dir = args.saveModel_dir
    batch_size = args.batch_size
    train_num = 2796
    writer1 = SummaryWriter(log_dir="log/loss")
    # ------- 1. define model --------
    # define the net

    net = PreMask().to(device)
    net.apply(weights_init_xavier)
    # define the loss

    criterion = LpLssimLoss().to(args.device)
    L1 = torch.nn.L1Loss().to(args.device)
    MSE=torch.nn.MSELoss().to(args.device)
    # ------- 2. define optimizer --------
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    # ------- 3. training process --------
    # ite_num = 0
    running_loss = 0.0
    # ite_num4val = 0
    #
    log_dir = model_dir + "epoch_114_loss_3.143497.pth" ###若需从断点处继续，需要更改此文件为上一次的epoch
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        net.load_state_dict(checkpoint['state_dict'])
        # net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')
    #
    #
    for epoch in range(start_epoch, n_epochs):
    # for epoch in range(0, n_epochs):
        net.train()
        t1 = time.time()
        for i, (image1, image2, label) in enumerate(train_loader):
            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)
            # ite_num = ite_num + 1
            # ite_num4val = ite_num4val + 1
            out = net(image1, image2)
            loss = MSE(out, label)
            # l1 = L1(out, label)
            # mssim=1-ms_ssim( out, label, data_range=1, size_average=True)
            # ssim = criterion(out, label)
            # loss = 0.8 * ssim + 0.2 * l1
            # loss =ssim
            # if i==0:
            #     im1=image1[0].detach().cpu()
            #     im2=image2[0].detach().cpu()
            #     lab1=label[0].detach().cpu()
            #     out_image=out[0].detach().cpu()
            #     writer1.add_image('resutl', make_grid([im1,im2,lab1,out_image], nrow=5, padding=20, normalize=False, pad_value=1), epoch)
            #     writer1.add_image('F1_ori', make_grid(F1_ori[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
            #     writer1.add_image('F2_ori', make_grid(F2_ori[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
            #     writer1.add_image('F1_att', make_grid(F1_a[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
            #     writer1.add_image('F2_att', make_grid(F2_a[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
            # writer1.add_image('fusion', make_grid(fusion[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
            # writer1.add_image('F1_a', make_grid(F1_a[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
            # writer1.add_image('F3_a', make_grid(F3_a[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
            # writer1.add_image('atten_fusion', make_grid(atten_fusion[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            if (i + 1) % 8 == 0:
                print("[epoch: %3d/%3d, batch: %5d/%5d] train loss: %8f " % (
                    epoch + 1, n_epochs, (i + 1) * batch_size, train_num, loss.item()))
        writer1.add_scalar('训练损失', running_loss, epoch)

        if epoch % 2 == 0:
            check_point = {  # 'state_dict': net.state_dict(),
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch}
            checkpoint_path = model_dir + "epoch_%d_loss_%3f.pth" % (epoch, running_loss)
            torch.save(check_point, checkpoint_path)

        # torch.save(net.state_dict(), model_dir + "epoch_%d_loss_%3f.pth" % (epoch, running_loss))
        running_loss = 0.0
        scheduler.step()
        t2 = time.time()
        print(t2 - t1)

    print('-------------Congratulations! Training Done!!!-------------')


def parse_args():
    parser = argparse.ArgumentParser()  # 创建一个解析对象
    parser.add_argument("--in_ch", type=int, default=3, help='rgb is 3,gray is 1')  # 向该对象中添加你要关注的命令行参数和选项
    parser.add_argument("--out_ch", type=int, default=64)  # 每一个add_argument方法对应一个你要关注的参数或选项
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=600, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=200, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='')
    parser.add_argument('--dataset_dir', type=str, default=r"D:/DeepLearing/set/set2")
    # parser.add_argument('--dataset_dir', type=str, default="DataSet/ori_data/DUTS-TR/trainData")
    parser.add_argument('--saveModel_dir', type=str, default=r'E:/MMAE/premask/')
    return parser.parse_args()  # 后调用parse_args()方法进行解析，解析成功之后即可使用


if __name__ == '__main__':
    training_setup_seed(1)
    args = parse_args()
    transforms_ = [transforms.Resize((256, 256), InterpolationMode.BICUBIC),
                   # transforms.RandomHorizontalFlip(p=0.6),
                   transforms.ToTensor()]
    train_set = SalObjDataset(dataset_dir=args.dataset_dir, transforms_=transforms_, rgb=True)
    salobj_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train(salobj_dataloader, args)

