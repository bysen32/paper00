import numpy as np
import scipy.misc
import os
from PIL import Image
from config import INPUT_SIZE, DEV_MODE
import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset
import core
from utils import transforms


class CUB:
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train

        img_name_list = []
        img_txt_file = open(os.path.join(self.root, "images.txt"))
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(" ")[-1])

        label_list = []
        label_txt_file = open(os.path.join(
            self.root, "image_class_labels.txt"))
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(" ")[-1]) - 1)

        train_test_list = []
        train_val_file = open(os.path.join(self.root, "train_test_split.txt"))
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(" ")[-1]))

        train_file_list = [x for i, x in zip(
            train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(
            train_test_list, img_name_list) if not i]

        if self.is_train:
            self.train_img = [scipy.misc.imread(os.path.join(
                self.root, "images", train_file)) for train_file in train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(
                train_test_list, label_list) if i][:data_len]
        else:
            self.test_img = [scipy.misc.imread(os.path.join(
                self.root, "images", test_file)) for test_file in test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(
                train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode="RGB")
            img = transforms.Resize((256, 256), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [
                                       0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode="RGB")
            img = transforms.Resize((256, 256), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [
                                       0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


TRAIN_DATASET = "train.txt"
EVAL_DATASET = "val.txt"

if DEV_MODE:
    TRAIN_DATASET = "train_dev.txt"
    EVAL_DATASET = "val_dev.txt"


def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB', INPUT_SIZE, 'white')
    return img


class CUB_Train:
    def __init__(self, root=None, dataloader=default_loader):
        self.transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        self.dataloader = dataloader

        self.root = root
        self.imgs = []
        self.labels = []

        with open(os.path.join(self.root, TRAIN_DATASET), 'r') as fid:
            for line in fid.readlines():
                img_path, label = line.strip().split()
                img = self.dataloader(img_path)
                label = int(label)
                self.imgs.append(img)
                self.labels.append(label)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img1 = self.transform(img)
        img2 = self.transform(img)
        img = torch.cat([img1, img2], dim=0)
        label = torch.cat([label, label], dim=0)
        index = torch.cat([index, index], dim=0)

        return [img, label, index]

    def __len__(self):
        return len(self.labels)


class CUB_Val:
    def __init__(self, root=None, dataloader=default_loader):
        self.transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        self.dataloader = dataloader

        self.root = root
        self.imgs = []
        self.labels = []

        with open(os.path.join(self.root, TRAIN_DATASET), 'r') as fid:
            for line in fid.readlines():
                img_path, label = line.strip().split()
                img = self.dataloader(img_path)
                label = int(label)
                self.imgs.append(img)
                self.labels.append(label)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img = self.transform(img)

        return [img, label, index]

    def __len__(self):
        return len(self.labels)


class CUB_Test:
    def __init__(self, root=None, dataloader=default_loader):
        self.transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        self.dataloader = dataloader

        self.root = root
        self.imgs = []
        self.labels = []

        with open(os.path.join(self.root, EVAL_DATASET), 'r') as fid:
            for line in fid.readlines():
                img_path, label = line.strip().split()
                img = self.dataloader(img_path)
                label = int(label)
                self.imgs.append(img)
                self.labels.append(label)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img = self.transform(img)

        return [img, label, index]

    def __len__(self):
        return len(self.labels)


class BatchDataset(Dataset):
    def __init__(self, root=None, dataloader=default_loader):
        self.transform1 = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.Resize([256, 256]),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(
                0.5, 1.5), saturation=(0.5, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(probability=0.5, sh=0.1)
        ])
        # 增强方法2： 关注更小的区域
        self.transform2 = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.Resize([336, 336]),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(
                0.5, 1.5), saturation=(0.5, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(probability=0.5, sh=0.1)
        ])
        self.dataloader = dataloader

        self.root = root
        with open(os.path.join(self.root, TRAIN_DATASET), 'r') as fid:
            self.imglist = fid.readlines()

        self.labels = []
        for line in self.imglist:
            image_path, label = line.strip().split()
            self.labels.append(int(label))
        self.labels = np.array(self.labels)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, index):
        image_name, label = self.imglist[index].strip().split()
        image_path = image_name
        img = self.dataloader(image_path)
        img1 = self.transform1(img)
        img2 = self.transform2(img)
        # images = torch.cat([img1, img2], dim=0)

        # label = torch.LongTensor([int(label)])
        label = int(label)
        # index = torch.LongTensor([index])

        return [(img1, img2), label, index]

    def __len__(self):
        return len(self.imglist)


# 修改这个取数据类的逻辑
# 1. {}
# 取到某个样本classA idx_a, -> 与该样本最相似的异类样本classB idx_b。
# 每个epoch将生成绝大部分的样本特征。
# 结束后，更新最近映射表
# 策略：找到很多环。取出打上使用标记。

class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        if False and len(core.model.g_InterPairs) == len(self.dataset):
            self.count = 0
            # while self.count + self.batch_size <= len(self.dataset):
            #     self.idxs_used = [False for _ in range(len(self.labels))]
            #     idxlist = [i for i in range(len(self.labels))]
            #     np.random.shuffle(idxlist)
            #     cur_idx = idxlist.pop()
            #     indices = []
            #     while len(indices) < self.batch_size:  # 随机取一个batch 可以重复
            #         if self.idxs_used[cur_idx]:
            #             cur_idx = idxlist.pop()
            #         self.idxs_used[cur_idx] = True
            #         indices.append(cur_idx)
            #         cur_idx = core.model.g_InterPairs[cur_idx][1].item()
            #     yield indices
            #     self.count += self.batch_size
            while self.count + self.batch_size <= len(self.dataset):
                classes = np.random.choice(self.labels_set, 1)
                indices = torch.sort(
                    core.model.g_LabelDiff[classes[0]], descending=True)[1]
                classes = np.append(classes, indices[:self.n_classes-1])
                indices = []
                for class_ in classes:
                    cur = self.used_label_indices_count[class_]
                    indices.extend(
                        self.label_to_indices[class_][cur: cur + self.n_samples])
                    self.used_label_indices_count[class_] += self.n_samples
                    if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                        np.random.shuffle(self.label_to_indices[class_])
                        self.used_label_indices_count[class_] = 0
                yield indices
                self.count += self.n_classes * self.n_samples
        else:
            self.count = 0
            while self.count + self.batch_size <= len(self.dataset):
                classes = np.random.choice(
                    self.labels_set, self.n_classes, replace=False)
                indices = []
                for class_ in classes:
                    cur = self.used_label_indices_count[class_]
                    indices.extend(
                        self.label_to_indices[class_][cur:cur+self.n_samples])
                    self.used_label_indices_count[class_] += self.n_samples
                    if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                        np.random.shuffle(self.label_to_indices[class_])
                        self.used_label_indices_count[class_] = 0
                yield indices
                self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


if __name__ == "__main__":
    trainset = CUB(root="./CUB_200_2011", is_train=True)
    print(len(trainset.train_img))
    print(len(trainset.train_label))
    for data in trainset:
        print(data[0].size(), data[1])

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=4, drop_last=False)
    for i, data in enumerate(trainloader):
        img, label = data[0].cuda(), data[1].cuda()
        print(i, label)

    # dataset = CUB(root="./CUB_200_2011", is_train=False)
    # print(len(dataset.test_img))
    # print(len(dataset.test_label))
    # for data in dataset:
    #     print(data[0].size(), data[1])

    testset = CUB_Test(root="./CUB_200_2011")
    print(len(testset.imgs))
    print(len(testset.labels))
    for data in testset:
        print(data[0].size(), data[1])

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=True, num_workers=4, drop_last=False)
    for i, data in enumerate(testloader):
        img, label = data[0].cuda(), data[1].cuda()
        print(i, label)
