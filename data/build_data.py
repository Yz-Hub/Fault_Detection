import os
import json
import torch
from monai.transforms import *
from monai.data import load_decathlon_datalist, Dataset, DataLoader


def build_loader(config):
    """
    构建训练和验证数据加载器。
    """
    data_dir = config.data.json_path
    datalist_json = os.path.join(data_dir, config.data.json_name)
    
    # 训练数据预处理
    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            EnsureChannelFirstd(keys=["label"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.9, 1.1)),
            RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(0, 1)),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=[0, 1]),
            RandRotated(keys=["image", "label"], range_x=0.25, range_y=0.25, range_z=0.0, mode="bilinear", prob=0.2, padding_mode="zeros"),
            RandGaussianSmoothd(keys=["image"], prob=0.1),
            RandGaussianNoised(keys=["image"], std=0.03, prob=0.2),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                spatial_size=(config.data.img_size, config.data.img_size, config.data.img_size),
                label_key="label",
                image_key="image",
                image_threshold=0.0,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    # 验证数据预处理
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            EnsureChannelFirstd(keys=["label"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                spatial_size=(config.data.img_size, config.data.img_size, config.data.img_size),
                label_key="label",
                image_key="image",
                image_threshold=0.0,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # 加载训练数据集
    datalist = load_decathlon_datalist(datalist_json, True, "training")
    train_ds = Dataset(data=datalist, transform=train_transform)
    
    # 训练数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,  # 训练集启用 shuffle
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=True,
    )

    # 加载验证数据集
    val_files = load_decathlon_datalist(datalist_json, True, "validation")
    val_ds = Dataset(data=val_files, transform=val_transform)
    
    # 验证数据加载器
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # 验证集 batch_size 通常为 1
        shuffle=False,  # 验证集不需要 shuffle
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=True,
    )

    return train_loader, val_loader


def create_json_files(foldername):
    """
    创建测试集的 JSON 文件。
    """
    imagepath = foldername  # 图像路径
    images = os.listdir(imagepath)
    res = {"testing": []}
    template = {"image": ""}
    
    for image in images:
        temp = template.copy()
        temp["image"] = os.path.join(imagepath, image)  # 拼接完整路径
        res["testing"].append(temp)
    
    # 保存 JSON 文件
    os.makedirs("datasets/test", exist_ok=True)
    json_path = os.path.join("datasets/test", "datas.json")
    with open(json_path, "w") as f:
        f.write(json.dumps(res))
    
    return json_path


def get_test_loader(json_dir):
    """
    构建测试数据加载器。
    """
    json_name = create_json_files(json_dir)
    datalist_json = json_name  # JSON 文件路径
    
    # 测试数据预处理
    test_transform = Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            ToTensord(keys=["image"]),
        ]
    )
    
    # 加载测试数据集
    test_files = load_decathlon_datalist(datalist_json, True, "testing")
    test_ds = Dataset(data=test_files, transform=test_transform)
    
    # 测试数据加载器
    test_loader = DataLoader(
        test_ds,
        batch_size=1,  # 测试集 batch_size 通常为 1
        shuffle=False,  # 测试集不需要 shuffle
        num_workers=1,  # 测试集通常不需要多线程
        pin_memory=False,  # 测试集通常不需要 pin_memory
        persistent_workers=True,
    )
    
    return test_loader