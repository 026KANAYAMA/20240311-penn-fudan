"""
モデルを学習、または推論するためのステップを定義
"""
import torch

def train_step(model, images, targets, optimizer, device):
    """
    学習ステップ
    """
    model.train()
    images = list(image.to(device) for image in images)

    #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    
    return losses.item()

def val_step(model, images, device):
    """
    検証ステップ
    """
    model.eval()
    images = list(image.to(device) for image in images)
    predictions = model(images)
    pred = predictions[0]
    
    return pred
