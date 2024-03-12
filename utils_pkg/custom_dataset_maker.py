"""
custom_dataset_maker.py : カスタムデータセットを作成するスクリプト
"""

import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

import unittest

class PennFudanDataset(torch.utils.data.Dataset):
    """
    custom dtatasetを作成する
    """
    def __init__(self, root, transforms):
        """
        PennFudanDatasetのコントラクタ
        
        Parameters:
        root (str): データセットのルーティング、この下に画像とマスクにある。
        transforms (callable): 画像とターゲットに適応される変換。
        """
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        """
        データセットから特定のインデックスに対応する画像とそのマスクを取得
        
        Parameters:
        idx (int): 取得したい画像とマスクのインデックス
        """
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones(num_objs, dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        """
        データセット内の画像の総数を返す。
        
        Parameters:
        """
        return len(self.imgs)


if __name__ == "__main__":
    class TestPennFudanDataset(unittest.TestCase):
        """
        PennFudanDatasetのテストクラス
        """
        def setUp(self):
            """
            テスト前のセットアップ
            
            parameters:
            """
            self.root = "../data/PennFudanPed"
            self.transforms = None
            self.dataset = PennFudanDataset(self.root, self.transforms)

        def test_getitem(self):
            """
            PennFudanDatasetの__getitem__メソッドのテスト

            Parameters:
            """
            idx = 0
            img, target = self.dataset.__getitem__(idx)
            # 適切な型であることを確認
            self.assertIsInstance(img, tv_tensors.Image)
            self.assertIsInstance(target, dict)
            self.assertIsInstance(target["boxes"], tv_tensors.BoundingBoxes)
            self.assertIsInstance(target["masks"], tv_tensors.Mask)
            self.assertIsInstance(target["labels"], torch.Tensor)
            self.assertIsInstance(target["image_id"], int)
            self.assertIsInstance(target["area"], torch.Tensor)
            self.assertIsInstance(target["iscrowd"], torch.Tensor)
            # 内容を確認
            print("img:", img)
            print("target:", target)
            

        def test_len(self):
            """
            PennFudanDatasetの__len__メソッドのテスト
            
            Parameters:
            """
            self.assertIsInstance(self.dataset.__len__(), int)
            # 内容を確認
            print("len:", self.dataset.__len__())
    
    unittest.main()