import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
path = r"D:\DeepLearing\pythonProject\code\result\Premask\color_lytro_20.png"
img = Image.open(path)
image = np.array(img).astype(np.float32)

pre_mask = transforms.ToTensor()(img)
pre_mask = torch.mean(pre_mask, dim=0, keepdim=True)
print(pre_mask)
a = torch.max(pre_mask)
print(a)
a = torch.min(pre_mask)
print(a)

print("完毕")