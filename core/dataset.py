import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE, DEV_MODE
from torch.utils.data.sampler import BatchSampler


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
        return Image.new('RGB', (224, 224), 'white')
    return img


class CUB_Train:
    def __init__(self, root=None, dataloader=default_loader):
        self.transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomCrop([244, 244]),
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
        img = (img1, img2)
        # label = torch.LongTensor([label])

        return [img, label]

    def __len__(self):
        return len(self.labels)


class CUB_Test:
    def __init__(self, root=None, dataloader=default_loader):
        self.transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomCrop([244, 244]),
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
        # label = torch.LongTensor([label])

        return [img, label]

    def __len__(self):
        return len(self.labels)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[
            0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(
                self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][self.used_label_indices_count[class_]])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size


if __name__ == "__main__":
    import torch
    # INPUT_SIZE = (448, 448)
    INPUT_SIZE = (224, 224)

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
