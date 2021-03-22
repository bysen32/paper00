import os
import sys
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, SAVE_FREQ, LR, resume, save_dir, WD, DEV_MODE
from core import model, dataset, resnet
from core.utils import init_log, progress_bar

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
if os.path.exists(save_dir):
    raise NameError("model dir exists!")

if DEV_MODE:
    _print = print
else:
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

# read dataset
# trainset = dataset.CUB(root="./CUB_200_2011", is_train=True, data_len=None)
trainset = dataset.CUB_Train(root="./CUB_200_2011")
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False)

# testset = dataset.CUB(root="./CUB_200_2011", is_train=False, data_len=None)
testset = dataset.CUB_Test(root="./CUB_200_2011")
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=False)

# define model
net = model.MyNet()
# net.avgpool = torch.nn.AdaptiveAvgPool2d(1)
# net.fc = torch.nn.Linear(512 * 4, 200)

# load saved model
if resume:
    ckpt = torch.load(resume)
    net.load_State_dict(ckpt["net_state_dict"])
    start_epoch = ckpt["epoch"] + 1
creterion = torch.nn.CrossEntropyLoss()

# define optimizers
raw_parameters = list(net.parameters())
raw_optimizer = torch.optim.SGD(
    raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)

schedulers = [
    MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
]

net = net.cuda()
net = DataParallel(net)

for epoch in range(start_epoch, 500):

    _print("--" * 50)
    net.train()
    for i, data in enumerate(trainloader):
        imgs, label = data[0], data[1].cuda()
        images1, images2 = imgs
        images1 = images1.cuda()
        images2 = images2.cuda()
        batch_size = images1.size(0)

        raw_optimizer.zero_grad()

        raw_logits, _, raw_features = net((images1, images2))
        raw_logits1, raw_logits2 = raw_logits
        raw_features1, raw_features2 = raw_features
        raw_loss1 = creterion(raw_logits1, label)
        raw_loss2 = creterion(raw_logits2, label)

        # target = torch.autograd.Variable(torch.ones(batch_size, 1)).cuda()
        # dist_loss = torch.nn.CosineEmbeddingLoss(reduction="mean")(raw_features1, raw_features2, target)
        dist_loss = torch.nn.L1Loss(reduction="mean")(
            raw_features1, raw_features2)
        # dist_loss = torch.nn.MSELoss(reduction="sum")(raw_features1, raw_features2)

        total_loss = raw_loss1 + raw_loss2 + dist_loss
        total_loss.backward()

        raw_optimizer.step()

        progress_bar(i, len(trainloader), "train")

    if epoch % SAVE_FREQ == 0:
        train_loss = 0
        train_correct1 = 0
        train_correct2 = 0
        total = 0
        net.eval()
        for i, data in enumerate(trainloader):
            with torch.no_grad():
                imgs, label = data[0], data[1].cuda()
                images1, images2 = imgs
                images1 = images1.cuda()
                images2 = images2.cuda()
                batch_size = images1.size(0)
                raw_logits, _, _ = net((images1, images2))
                raw_logits1, raw_logits2 = raw_logits
                # caculate loss
                raw_loss1 = creterion(raw_logits1, label)
                raw_loss2 = creterion(raw_logits2, label)
                # caculate accuracy
                _, raw_predict1 = torch.max(raw_logits1, 1)
                _, raw_predict2 = torch.max(raw_logits2, 1)
                total += batch_size * 2
                train_correct1 += torch.sum(raw_predict1.data == label.data)
                train_correct2 += torch.sum(raw_predict2.data == label.data)
                train_loss += raw_loss1.item() * batch_size + raw_loss2.item() * batch_size
                progress_bar(i, len(trainloader), "eval train set")

        train_acc = float(train_correct1 + train_correct2) / total
        train_loss = train_loss / total

        _print("epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}".format(
            epoch, train_loss, train_acc, total))

        # evaluate on test set
        test_loss = 0
        test_correct = 0
        total = 0
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                raw_logits, _, _ = net(img, flag="test")
                raw_loss = creterion(raw_logits, label)
                _, raw_predict = torch.max(raw_logits, 1)
                total += batch_size
                test_correct += torch.sum(raw_predict.data == label.data)
                test_loss += raw_loss.item() * batch_size
                progress_bar(i, len(testloader), "eval test set")

        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        _print("epoch:{} - test loss: {:.4f} and test acc: {:.3f} total sample: {}".format(
            epoch, test_loss, test_acc, total))

        net_state_dict = net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if not DEV_MODE:
            torch.save(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "net_state_dict": net_state_dict,
                },
                os.path.join(save_dir, "%03d.ckpt" % epoch),
            )

print("finishing training")
