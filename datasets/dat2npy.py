import os  # 用于处理文件和目录路径
from tqdm import tqdm  # 用于显示进度条
import numpy as np  # 用于处理数组和矩阵操作

# 定义一个函数，用于将.dat文件中的数据读取并转换为指定形状的numpy数组
def dat_npy(dat_file, shape=(128, 128, 128)):
    with open(dat_file, 'r') as f:  # 打开.dat文件
        data = np.fromfile(f, dtype=np.float32)  # 从文件中读取数据并转换为float32类型的numpy数组
        data = data.reshape(shape)  # 将数据重塑为指定的形状（默认为128x128x128）
    return data  # 返回处理后的numpy数组

# 创建训练和验证所需的目录结构
os.makedirs(r'200-20/train/images', exist_ok=True)  # 创建训练集图像目录，如果目录已存在则忽略
os.makedirs(r'200-20/train/labels', exist_ok=True)  # 创建训练集标签目录，如果目录已存在则忽略
os.makedirs(r'200-20/val/images', exist_ok=True)  # 创建验证集图像目录，如果目录已存在则忽略
os.makedirs(r'200-20/val/labels', exist_ok=True)  # 创建验证集标签目录，如果目录已存在则忽略

# 定义一个函数，用于将指定路径下的.dat文件转换为.npy文件并保存到相应的目录
def save(path, image_path, label_path):
    dat = []  # 用于存储.dat文件的路径
    list = os.listdir(path)  # 获取指定路径下的所有文件和目录
    for datname in list:  # 遍历目录中的每个文件
        dat_path = os.path.join(path, datname)  # 构建完整的文件路径
        dat.append(dat_path)  # 将文件路径添加到列表中

    # 使用tqdm显示进度条，遍历所有.dat文件
    for path in tqdm(dat):
        data = dat_npy(path, shape=(128, 128, 128))  # 将.dat文件转换为numpy数组
        name = path.split('\\')[-1].replace('.dat', '.npy')  # 获取文件名并将扩展名从.dat替换为.npy
        if int(name.split('.')[0]) < 200:  # 如果文件名中的数字小于200，则保存到图像目录
            np.save(os.path.join(image_path, name), data)  # 将numpy数组保存为.npy文件
        else:  # 否则保存到标签目录
            np.save(os.path.join(label_path, name), data)  # 将numpy数组保存为.npy文件

# 主程序入口
if __name__ == '__main__':
    seismic_path = r'seis'  # 地震数据文件的路径
    label_path = r'fault'  # 标签数据文件的路径
    train_image_path = r'200-20/train/images'  # 训练集图像保存路径
    train_label_path = r'200-20/train/labels'  # 训练集标签保存路径
    val_image_path = r'200-20/val/images'  # 验证集图像保存路径
    val_label_path = r'200-20/val/labels'  # 验证集标签保存路径

    # 调用save函数，将地震数据保存到训练集和验证集图像目录
    save(seismic_path, train_image_path, val_image_path)
    # 调用save函数，将标签数据保存到训练集和验证集标签目录
    save(label_path, train_label_path, val_label_path)