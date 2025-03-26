# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Tang
# --------------------------------------------------------

import os
import sys
import logging
import functools
from termcolor import colored
import torch.distributed as dist

@functools.lru_cache()
def create_logger(output_dir, name):
    """
    创建一个 logger 对象，用于记录日志。
    :param output_dir: 日志文件输出目录
    :param name: logger 名称
    """
    # 自动判断是否启用分布式训练
    if dist.is_available() and dist.is_initialized():
        # 分布式训练环境，获取当前 rank
        dist_rank = dist.get_rank()
    else:
        # 单 GPU 训练环境，rank 默认为 0
        dist_rank = 0

    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # 创建日志格式
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # 创建控制台 handler（仅在 dist_rank == 0 时输出到控制台）
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # 创建文件 handler
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger