import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

'''批量转换rgb为y并保存'''
if __name__ == '__main__':
    img_path = r"E:\Users\13687\PycharmProjects\GENFusion\Dataset\test\medical\PET"
    save_path = r"E:\Users\13687\PycharmProjects\SwinFusion-master\Dataset\testsets\PET-MRI\PET_Y"
    # opencv读取训练集源图像，转换为ycbcr空间并写入train_save_dir
    # for img in os.listdir(source_img_path):
    for img in os.listdir(img_path):
        source_img = cv2.imread(img_path + '\\' + img, cv2.COLOR_BGR2RGB)
        ycrcb_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2YCrCb)  # opencv默认只能变为ycrcb
        ycbcr_img = ycrcb_img[:, :, (0, 2, 1)]  # ycrcb变为ycbcr
        y = ycbcr_img[:, :, 0]  # 单独取出y亮度分量
        cv2.imwrite(save_path + '\\' + img, y)
        # pil读取图像并转换为ycbcr
        # img3 = Image.open(image_path)
        # ycbcr_img3 = img3.convert("YCbCr")

    trans = transforms.ToTensor()


def fused_to_y(fused_img):
    x = np.zeros((1, 1, 256, 256))
    fused_img_np = fused_img.cpu().detach().numpy()  # [5, 3, 256, 256]
    for i in range(fused_img_np.shape[0]):
        batchsize_i = fused_img_np[i, :, :, :]
        batchsize_i_hwc = np.transpose(batchsize_i, (1, 2, 0))  # [256,256,3]
        ycrcb = cv2.cvtColor(batchsize_i_hwc, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[:, :, 0][np.newaxis, np.newaxis, :]  # [1, 1, 256, 256]
        if i == 0:
            x = x + y

        if i != 0:
            x = np.concatenate((x, y), axis=0)
        # x = x[np.newaxis, :, :, :]
    return torch.from_numpy(x).float()


# 适用于h w c的rgb图像转为 y cb cr
# def rgb2ycbcr(img_rgb):
#     R = img_rgb[:, :, 0]
#     G = img_rgb[:, :, 1]
#     B = img_rgb[:, :, 2]
#     Y = 0.299 * R + 0.587 * G + 0.114 * B
#     Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255.0
#     Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255.0
#     # Y=0.257*R+0.564*G+0.098*B+16

#     # Cb=-0.148*R-0.291*G+0.439*B+128/255.0

#     # Cr=0.439*R-0.368*G-0.071*B+128/255.0
# return Y, Cb, Cr

def rgb2ycbcr(img_rgb):
    x = np.zeros_like(img_rgb[0, :, :, :][np.newaxis, :, :, :])
    for i in range(img_rgb.shape[0]):
        R = img_rgb[i, :, :, 0]
        G = img_rgb[i, :, :, 1]
        B = img_rgb[i, :, :, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255.0
        Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255.0

        Y = np.expand_dims(Y, axis=-1)
        Cb = np.expand_dims(Cb, axis=-1)
        Cr = np.expand_dims(Cr, axis=-1)
        YCbCr = np.concatenate([Y, Cb, Cr], axis=-1)[np.newaxis, :, :, :]
        # print(YCbCr.shape)
        if i == 0:
            x = x + YCbCr
            # print(x.shape)
        if i != 0: 
            x = np.concatenate([x, YCbCr], axis=0)
            # print(x.shape)
        # print(i)
    return x


# 适用于y cb cr的转rgb h w c
def ycbcr2rgb(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128/255.0)
    G = Y - 0.34414 * (Cb - 128/255.0) - 0.71414 * (Cr - 128/255.0)
    B = Y + 1.772 * (Cb - 128/255.0)
    # R = 1.164*(Y-16)+1.596*(Cr-128/255.0)
    # G = 1.164*(Y-16)-0.392*(Cb-128/255.0)-0.813*(Cr-128/255.0)
    # B = 1.164*(Y-16)+2.817*(Cb-128/255.0)
    
    R = np.expand_dims(R, axis=-1)
    G = np.expand_dims(G, axis=-1)
    B = np.expand_dims(B, axis=-1)
    return np.concatenate([B, G, R], axis=-1)


def fusecbcr(img1cb, img2cb):
    a = abs(img1cb - 128/255.0)
    b = abs(img2cb - 128/255.0)
    fusedcbcr = (img1cb * a + img2cb * b) / (a + b)
    return fusedcbcr