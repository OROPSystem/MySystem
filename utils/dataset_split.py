# --------------------------------------
# 1. Dataset dir split train/valid/test
# 2. Build Dataset
# 3. Build DataLoader
# --------------------------------------

import os
import shutil
import random
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


class DatasetSplit:
    def __init__(self, user_name, dataset_name, ):
        self.user_name = user_name
        self.dataset_name = dataset_name

    def split_dataset(self, split_ratio=(0.8, 0.1, 0.1)):
        assert sum(split_ratio) == 1, "split ratio define should like (train, valid, test), please check it!"
        img_train_root = "./user_data/" + self.user_name + "/datasets/image/" + self.dataset_name + "/train"
        img_valid_root = "./user_data/" + self.user_name + "/datasets/image/" + self.dataset_name + "/valid"
        img_test_root = "./user_data/" + self.user_name + "/datasets/image/" + self.dataset_name + "/test"
        os.makedirs(img_train_root, exist_ok=True)
        os.makedirs(img_valid_root, exist_ok=True)
        os.makedirs(img_test_root, exist_ok=True)

        shutil.copytree("./user_data/" + self.user_name + "/datasets/image/" + self.dataset_name + "/train_origin", img_train_root, dirs_exist_ok=True)

        cls = os.listdir(img_train_root)
        for c in cls:
            cls_train_root = os.path.join(img_train_root, c)
            cls_images = os.listdir(cls_train_root)
            n = len(cls_images)
            n_val = int(n * split_ratio[1])
            n_tes = int(n * split_ratio[2])

            # split valid dataset
            valid_samples = random.sample(cls_images, n_val)
            valid_root = img_valid_root + '/' + c
            os.makedirs(valid_root, exist_ok=True)
            for v in valid_samples:
                shutil.move(cls_train_root + '/' + v, valid_root + '/' + v)

            # split test dataset
            cls_images = os.listdir(cls_train_root)
            test_samples = random.sample(cls_images, n_tes)
            test_root = img_test_root + '/' + c
            os.makedirs(test_root, exist_ok=True)
            for t in test_samples:
                shutil.move(cls_train_root + '/' + t, test_root + '/' + t)

    def create_dataloader_iterator(self, mode, batch_size, trans=None, shuffle=True, drop_last=False):
        img_path = f"./user_data/{self.user_name}/datasets/image/{self.dataset_name}/{mode}"
        dataset = datasets.ImageFolder(img_path, transform=trans)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return dataloader


if __name__ == '__main__':
    # trans = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ])
    # dl = create_dataloader_iterator("../user_data/test/datasets/image/NEU_50/train", 32, trans=trans)
    # for i, img in enumerate(dl):
    #     if i == 0:
    #         print(img[0].shape, img[1])
    print(os.getcwd())
