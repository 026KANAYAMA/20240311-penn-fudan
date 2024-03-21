"""
学習、推論の実行、ヘルパー関数を利用した簡易版
"""
import torch

from step import train_step, val_step
from references.detection.engine import train_one_epoch, evaluate
from references.detection import utils, coco_eval, coco_utils
from utils_pkg import PennFudanDataset, get_transform, get_model_instance_segmentation

def worker_init_fn(worker_id):
    torch.manual_seed(1 + worker_id)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = PennFudanDataset("data/PennFudanPed", get_transform(train=True))
dataset_test = PennFudanDataset("data/PennFudanPed", get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn, worker_init_fn=worker_init_fn 
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn, worker_init_fn=worker_init_fn 
)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10
hist = {'loss':[], 'val_loss':[]}

for epoch in range(num_epochs):
    train_loss = 0

    # train on the training dataset
    for images, targets in data_loader:
        train_loss += train_step(model, images, targets, optimizer, device)

    train_loss /= len(data_loader)
    hist['loss'].append(train_loss)
    print(f"Epoch {epoch+1}: train_loss = {train_loss}")

    # update the learning rate
    lr_scheduler.step()

print("That's it!")

# save the model
torch.save(model.state_dict(), "results/mask_r_cnn.pth")