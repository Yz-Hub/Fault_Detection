# -*- coding: utf-8 -*-
import os
import json

def tojson(path):
    # 将传入的路径转换为绝对路径
    path = os.path.abspath(path)
    print(f"处理路径: {path}")  # 打印路径以调试

    # 使用 os.path.join 拼接路径
    imagepath = os.path.join("train", "images")  # 修改为 os.path.join
    labelpath = os.path.join("train", "labels")  # 修改为 os.path.join
    images_dir = os.path.join(path, imagepath)  # 完整路径
    print(f"图像目录: {images_dir}")  # 打印路径以调试

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")

    images = os.listdir(images_dir)  # 修改为 os.path.join
    res = {"training": [], "validation": []}
    template = {"image": "", "label": ""}
    for image in images:
        temp = template.copy()
        temp["image"] = os.path.join(path, imagepath, image)  # 修改为 os.path.join
        temp["label"] = os.path.join(path, labelpath, image)  # 修改为 os.path.join
        res["training"].append(temp)

    # 验证集路径
    imagepath = os.path.join("val", "images")  # 修改为 os.path.join
    labelpath = os.path.join("val", "labels")  # 修改为 os.path.join
    images_dir = os.path.join(path, imagepath)  # 完整路径
    print(f"验证图像目录: {images_dir}")  # 打印路径以调试

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"验证图像目录不存在: {images_dir}")

    images = os.listdir(images_dir)  # 修改为 os.path.join
    for image in images:
        temp = template.copy()
        temp["image"] = os.path.join(path, imagepath, image)  # 修改为 os.path.join
        temp["label"] = os.path.join(path, labelpath, image)  # 修改为 os.path.join
        res["validation"].append(temp)

    # 输出文件路径
    output_path = os.path.join('datasets', 'dataset.json')  # 修改为 os.path.join
    print(f"输出文件路径: {output_path}")  # 打印路径以调试
    with open(output_path, 'w') as f:
        f.write(json.dumps(res))