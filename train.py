import os
import sys
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, SAVE_FREQ, LR, resume, save_dir, WD, DEV_MODE, VERSION_HEAD
from core import model, dataset, resnet
from core.utils import init_log, progress_bar

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
start_epoch = 1
best_acc = 0
save_dir = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
if os.path.exists(save_dir):
    raise NameError("model dir exists!")

os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info


# read dataset
# trainset = dataset.CUB(root="./CUB_200_2011", is_train=True, data_len=None)
trainset = dataset.CUB_Train(root="./CUB_200_2011")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False)

# testset = dataset.CUB(root="./CUB_200_2011", is_train=False, data_len=None)
testset = dataset.CUB_Test(root="./CUB_200_2011")
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8, drop_last=False)

# define model
net = model.MyNet()
# net.avgpool = torch.nn.AdaptiveAvgPool2d(1)
# net.fc = torch.nn.Linear(512 * 4, 200)

# load saved model
if resume and os.path.isfile(resume):
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt["net_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    _print("Load {}".format(resume))

criterion = torch.nn.CrossEntropyLoss().cuda()

# define optimizers
raw_parameters = list(net.parameters())
raw_optimizer = torch.optim.SGD(
    raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)

schedulers = [
    MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
]

# ----------------- pairs struct -----------------
rank_criterion = torch.nn.MarginRankingLoss(margin=0.05)
softmax_layer = torch.nn.Softmax(dim=1).cuda()

net = net.cuda()
net = DataParallel(net)

_print("-"*10+VERSION_HEAD+"-"*10)

for epoch in range(start_epoch, 500):

    _print("--" * 50)
    # _print("resnet50 model1: cossim model2 224x224 batchsize:30")
    net.train()
    for i, data in enumerate(trainloader):
        imgs, labels = data[0], data[1].cuda()
        images1, images2 = imgs
        images1 = images1.cuda()
        images2 = images2.cuda()
        batch_size = images1.size(0)

        raw_optimizer.zero_grad()

        raw_logits, _, projected_features, pairs_logits, pairs_labels = net((images1, images2), labels)
        raw_logits1, raw_logits2 = raw_logits
        raw_loss1 = criterion(raw_logits1, labels)
        raw_loss2 = criterion(raw_logits2, labels)

        projected_features1, projected_features2 = projected_features
        # resnet50 224x224 lambda=1 project+cosine_similarity acc:80.0%
        target = torch.autograd.Variable(torch.ones(batch_size, 1)).cuda()
        dist_loss = torch.nn.CosineEmbeddingLoss(reduction="mean")(
            projected_features1, projected_features2, target)
        # dist_loss = torch.nn.MSELoss(reduction="mean")(projected_features1, projected_features2)
        # dist_loss = torch.nn.L1Loss(reduction="mean")(projected_features1, projected_features2)

        # ---------- pairs attention struct ---------------------
        logit1_self, logit1_other, logit2_self, logit2_other = pairs_logits
        # ??? 翻倍了
        batch_size = logit1_self.shape[0]
        labels1, labels2 = pairs_labels
        labels1 = labels1.cuda()
        labels2 = labels2.cuda()
        self_logits = torch.zeros(2*batch_size, 200).cuda()
        other_logits = torch.zeros(2*batch_size, 200).cuda()
        self_logits[:batch_size] = logit1_self
        self_logits[batch_size:] = logit2_self
        other_logits[:batch_size] = logit1_other
        other_logits[batch_size:] = logit2_other

        logits = torch.cat([self_logits, other_logits], dim=0)
        targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)
        softmax_loss = criterion(logits, targets)

        self_scores = softmax_layer(self_logits)[torch.arange(2*batch_size).cuda().long(), torch.cat([labels1, labels2], dim=0)]
        other_scores = softmax_layer(other_logits)[torch.arange(2*batch_size).cuda().long(), torch.cat([labels1, labels2], dim=0)]
        flag = torch.ones([2*batch_size, ]).cuda()
        rank_loss = rank_criterion(self_scores, other_scores, flag)

        total_loss = raw_loss1 + raw_loss2
        if epoch > 20:
            total_loss += dist_loss
        if epoch > 50:
            total_loss += softmax_loss + rank_loss
        total_loss.backward()

        raw_optimizer.step()
        for sch in schedulers:
            sch.step()

        progress_bar(i, len(trainloader), "train")

    if epoch % SAVE_FREQ == 0:
        train_loss = 0
        train_correct1 = 0
        train_correct2 = 0
        total = 0
        net.eval()
        for i, data in enumerate(trainloader):
            with torch.no_grad():
                imgs, labels = data[0], data[1].cuda()
                images1, images2 = imgs
                images1 = images1.cuda()
                images2 = images2.cuda()
                batch_size = images1.size(0)

                raw_logits, _, _, _, _ = net((images1, images2), labels)
                raw_logits1, raw_logits2 = raw_logits
                # caculate class loss 
                raw_loss1 = criterion(raw_logits1, labels)
                raw_loss2 = criterion(raw_logits2, labels)

                # caculate accuracy
                _, raw_predict1 = torch.max(raw_logits1, 1)
                _, raw_predict2 = torch.max(raw_logits2, 1)
                total += batch_size * 2
                train_correct1 += torch.sum(raw_predict1.data == labels.data)
                train_correct2 += torch.sum(raw_predict2.data == labels.data)
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
                img, labels = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)

                raw_logits, _, _ = net(img, labels, flag="test")
                # caculate class loss
                raw_loss = criterion(raw_logits, labels)

                # caculate accuracy
                _, raw_predict = torch.max(raw_logits, 1)
                total += batch_size
                test_correct += torch.sum(raw_predict.data == labels.data)
                test_loss += raw_loss.item() * batch_size
                progress_bar(i, len(testloader), "eval test set")

        test_acc = float(test_correct) / total
        if (best_acc < test_acc):
            best_acc = test_acc
        test_loss = test_loss / total
        _print("epoch:{} - test loss: {:.4f} and test acc: {:.3f}/{:.3f} total sample: {}".format(
            epoch, test_loss, test_acc, best_acc, total))

        net_state_dict = net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # 暂时不保存模型
        torch.save(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "net_state_dict": net_state_dict,
            },
            "checkpoint.ckpt"
        )

print("finishing training")
