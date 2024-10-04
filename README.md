# MMAE
This is the code for article MMAE: A universal image fusion method via mask attention mechanism.
The detailed code has been uploaded.

这是文章 MMAE: A universal image fusion method via mask attention mechanism 的代码。
详细代码已经上传。

# 修改日期 2024.07.05 Date of revision 2024.07.05

# 数据集 Datasets
## 训练数据集 Training Datasets
link 链接：https://pan.baidu.com/s/1up-FGeMLt0_LQACDWjDYoQ?pwd=6356 

extraction code 提取码：6356

## 测试数据集 Test Datasets
link 链接：https://pan.baidu.com/s/1cP7DKddQHjxO6W6ABZc7cg?pwd=hwso 

extraction code 提取码：hwso

# 使用方法 How to use
## 训练方法，需要根据实际情况修改train文件中数据集地址和训练参数的保存地址，数据集读取可以自行根据实际情况修改，此处提供了一种参考
Training method: you need to modify the dataset address in the TRAIN file and the save address of the training parameters according to the actual situation. The dataset reading can be modified by yourself according to the actual situation. A reference is provided here.

多聚焦图像融合任务的训练，运行train.py

红外和可见光图像融合任务的训练，运行train-5.py

医学图像融合任务的训练，运行train-6.py

Training for the multifocus image fusion task, running train.py

Training for infrared and visible image fusion task, running train-5.py

Training for medical image fusion task, running train-6.py

## 测试方法，需要根据实际情况修改test文件中训练参数的读取地址和融合图像保存地址
Test method: you need to modify the reading address of the training parameters and the fusion image saving address in the TEST file according to the actual situation.

多聚焦图像融合任务测试，运行test.py

红外和可见光图像融合任务测试，运行test-5.py

医学图像融合任务测试，运行test-6.py

Multi-focus image fusion task test, running test.py

Infrared and visible image fusion task test, running test-5.py

Medical image fusion task test, running test-6.py

## 保存的训练文件下载地址
Download address for saved training files

link 链接：https://pan.baidu.com/s/1pOE-7MzQ8KWa5NGgwz9AfA?pwd=yjig 

extraction code 提取码：yjig 

或者直接使用pth文件夹中的epoch_82_loss_6.166936.pth文件

Or just use the epoch_82_loss_6.166936.pth file in the pth folder

# 关于引用 About the citation
文章已上线，欢迎引用。
The article is now online. Feel free to cite it.

Article Link 文章链接: https://www.sciencedirect.com/science/article/abs/pii/S0031320324007921

@article{WANG2025111041,
title = {MMAE: A universal image fusion method via mask attention mechanism},
journal = {Pattern Recognition},
volume = {158},
pages = {111041},
year = {2025},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2024.111041},
url = {https://www.sciencedirect.com/science/article/pii/S0031320324007921},
author = {Xiangxiang Wang and Lixing Fang and Junli Zhao and Zhenkuan Pan and Hui Li and Yi Li}
}

