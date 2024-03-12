"""
for_transformation.py: 画像変換を行う関数を定義するスクリプト
"""

from torchvision.transforms import v2 as T
import torch

def get_transform(train):
    """
    transformを適用する
    
    Parameters:
    train : bool
        学習時にはTrue、それ以外はFalse
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

if __name__ == "__main__":
    get_transform(True)
