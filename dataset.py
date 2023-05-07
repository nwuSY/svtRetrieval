import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(256),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]
)

test_valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])


class SV1KDataset(Dataset):
    def __init__(self, transform, gw_root_dir="./data", train=True):
        self.train = train
        self.gw_root_dir = gw_root_dir
        self.transform = transform
        self.title_list = sorted([elem for elem in os.listdir(os.path.join(self.gw_root_dir, "text"))])

    def __len__(self):
        return len(self.title_list)

    def __getitem__(self, index):
        text_index = self.title_list[index]
        number = text_index.split(".")[0]

        if self.train == "train":
            path = os.path.join(self.gw_root_dir, "sv1k", "train", number + ".jpg")
            img = Image.open(path).convert("RGB")
            img_1, img_2 = self.transform(img), self.transform(img)
        elif self.train == "test":
            path = os.path.join(self.gw_root_dir, "sv1k", "test", number + ".png")
            img = Image.open(path).convert("RGB")
            img_1, img_2 = self.transform(img), self.transform(img)
        else:
            path = os.path.join(self.gw_root_dir, "sv1k", "valid", number + ".png")
            img = Image.open(path).convert("RGB")
            img_1, img_2 = self.transform(img), self.transform(img)

        text_features = torch.load(os.path.join(self.gw_root_dir, "text", number + ".pkl"))  # shape:[15, 604]

        return number, img_1, img_2, text_features
