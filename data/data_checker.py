"""
data_checker.py : 画像とセグメンテーションマスクのペアを1組確認するスクリプト
"""

import matplotlib.pyplot as plt
from torchvision.io import read_image
import os

image = read_image("data/PennFudanPed/PNGImages/FudanPed00001.png")
mask = read_image("data/PennFudanPed/PedMasks/FudanPed00001_mask.png")

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title("Image")
plt.imshow(image.permute(1, 2, 0)) #channel first表現をchannel last表現に変換
plt.subplot(122)
plt.title("Mask")
plt.imshow(mask.permute(1, 2, 0))

if not os.path.exists("results"):
    os.makedirs("results")
plt.savefig("results/sample_1.png")